/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.ui.llmchat

import com.google.ai.edge.litertlm.Capabilities
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.common.cleanUpMediapipeTaskErrorMessage
import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.DEFAULT_TEMPERATURE
import com.google.ai.edge.gallery.data.DEFAULT_TOPK
import com.google.ai.edge.gallery.data.DEFAULT_TOPP
import com.google.ai.edge.gallery.data.DEFAULT_VISION_ACCELERATOR
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.ModelCapability
import com.google.ai.edge.gallery.data.THOUGHT_CHANNEL
import com.google.ai.edge.gallery.data.markInitializationFailed
import com.google.ai.edge.gallery.data.markInitializationStarted
import com.google.ai.edge.gallery.data.markInitialized
import com.google.ai.edge.gallery.data.resetInitialization
import com.google.ai.edge.gallery.runtime.CleanUpListener
import com.google.ai.edge.gallery.runtime.LlmModelHelper
import com.google.ai.edge.gallery.runtime.ResultListener
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolProvider
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CancellationException
import kotlinx.coroutines.CoroutineScope

private const val TAG = "AGLlmChatModelHelper"

data class LlmModelInstance(val engine: Engine, var conversation: Conversation)

object LlmChatModelHelper : LlmModelHelper {
  // Indexed by model name.
  private val cleanUpListeners: MutableMap<String, CleanUpListener> = mutableMapOf()

  @OptIn(ExperimentalApi::class) // opt-in experimental flags
  override fun initialize(
    context: Context,
    model: Model,
    taskId: String,
    supportImage: Boolean,
    supportAudio: Boolean,
    onDone: (String) -> Unit,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    coroutineScope: CoroutineScope?,
  ) {
    if (model.instance != null) {
      Log.d(TAG, "Model '${model.name}' already initialized in LlmChatModelHelper. Skipping.")
      model.markInitialized()
      onDone("")
      return
    }
    model.markInitializationStarted()
    // Prepare options.
    val maxTokens =
      model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)
    val topK = model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
    val topP = model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
    val temperature =
      model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
    val accelerator =
      model.getStringConfigValue(key = ConfigKeys.ACCELERATOR, defaultValue = Accelerator.GPU.label)
    val visionAccelerator =
      model.getStringConfigValue(
        key = ConfigKeys.VISION_ACCELERATOR,
        defaultValue = DEFAULT_VISION_ACCELERATOR.label,
      )
    val shouldEnableImage = supportImage
    val shouldEnableAudio = supportAudio
    val optimalCpuThreads = maxOf(2, minOf(4, Runtime.getRuntime().availableProcessors()))
    val preferredBackend =
      when (accelerator) {
        Accelerator.CPU.label -> Backend.CPU(optimalCpuThreads)
        Accelerator.GPU.label -> Backend.GPU()
        Accelerator.NPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        Accelerator.TPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        else -> Backend.CPU(optimalCpuThreads)
      }
    Log.d(TAG, "Preferred backend: $preferredBackend (CPU threads: $optimalCpuThreads)")

    val visionBackend =
      when (visionAccelerator) {
        Accelerator.CPU.label -> Backend.CPU(optimalCpuThreads)
        Accelerator.GPU.label -> if (preferredBackend is Backend.CPU) Backend.CPU(optimalCpuThreads) else Backend.GPU()
        Accelerator.NPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        Accelerator.TPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        else -> if (preferredBackend is Backend.CPU) Backend.CPU(optimalCpuThreads) else Backend.GPU()
      }

    val modelPath = model.getPath(context = context)
    val cacheDir =
      if (modelPath.startsWith("/data/local/tmp")) {
        context.getExternalFilesDir(null)?.absolutePath
      } else {
        null
      }
    val engineConfig =
      EngineConfig(
        modelPath = modelPath,
        backend = preferredBackend,
        visionBackend = if (shouldEnableImage) visionBackend else null, // must be GPU for Gemma 3n
        audioBackend = if (shouldEnableAudio) Backend.CPU(optimalCpuThreads) else null, // must be CPU for Gemma 3n
        maxNumTokens = maxTokens,
        cacheDir = cacheDir,
      )

    // Check if the model file supports speculative decoding.
    var supportsSpeculativeDecoding = false
    // Check if the model file supports speculative decoding.
    try {
      Capabilities(modelPath).use {
        supportsSpeculativeDecoding = it.hasSpeculativeDecodingSupport()
      }
    } catch (e: Exception) {
      // Ignore exceptions and assume not supported.
    }
    // Create an instance of LiteRT LM engine and conversation.
    try {
      var speculativeDecoding = false
      // Check if the model supports speculative decoding for the given task type and if the
      // speculative decoding is enabled in the settings.
      if (
        supportsSpeculativeDecoding &&
          model.capabilityToTaskTypes[ModelCapability.SPECULATIVE_DECODING]?.contains(taskId) ==
            true
      ) {
        speculativeDecoding =
          model.getBooleanConfigValue(
            key = ConfigKeys.ENABLE_SPECULATIVE_DECODING,
            defaultValue = false,
          )
      }
      ExperimentalFlags.enableSpeculativeDecoding = speculativeDecoding
      Log.d(TAG, "Speculative decoding enabled: $speculativeDecoding")
      var activeEngine = Engine(engineConfig)
      try {
        activeEngine.initialize()
      } catch (e: Exception) {
        Log.w(TAG, "Backend initialization failed: ${e.message}. Attempting CPU fallback.")
        try {
          val fallbackConfig =
            engineConfig.copy(
              backend = Backend.CPU(optimalCpuThreads),
              visionBackend = if (shouldEnableImage) Backend.CPU(optimalCpuThreads) else null,
              audioBackend = if (shouldEnableAudio) Backend.CPU(optimalCpuThreads) else null,
            )
          activeEngine = Engine(fallbackConfig)
          activeEngine.initialize()
        } catch (e2: Exception) {
          if (shouldEnableImage || shouldEnableAudio) {
            Log.w(TAG, "Multimodal CPU initialization failed: ${e2.message}. Falling back to text-only CPU.")
            val textOnlyConfig =
              engineConfig.copy(
                backend = Backend.CPU(optimalCpuThreads),
                visionBackend = null,
                audioBackend = null,
              )
            activeEngine = Engine(textOnlyConfig)
            activeEngine.initialize()
          } else {
            throw e2
          }
        }
      }
      ExperimentalFlags.enableSpeculativeDecoding = false

      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      val conversation =
        try {
          activeEngine.createConversation(
            ConversationConfig(
              samplerConfig =
                if (preferredBackend is Backend.NPU) {
                  null
                } else {
                  SamplerConfig(
                    topK = topK,
                    topP = topP.toDouble(),
                    temperature = temperature.toDouble(),
                  )
                },
              systemInstruction = systemInstruction,
              tools = tools,
            )
          )
        } catch (e: Exception) {
          Log.w(TAG, "createConversation failed with constrained decoding/tools: ${e.message}. Retrying plain conversation.")
          ExperimentalFlags.enableConversationConstrainedDecoding = false
          activeEngine.createConversation(
            ConversationConfig(
              samplerConfig =
                if (preferredBackend is Backend.NPU) {
                  null
                } else {
                  SamplerConfig(
                    topK = topK,
                    topP = topP.toDouble(),
                    temperature = temperature.toDouble(),
                  )
                },
              systemInstruction = systemInstruction,
              tools = emptyList(),
            )
          )
        }
      ExperimentalFlags.enableConversationConstrainedDecoding = false
      model.instance = LlmModelInstance(engine = activeEngine, conversation = conversation)
    } catch (e: Exception) {
      val errorMsg = cleanUpMediapipeTaskErrorMessage(e.message ?: "Unknown error")
      model.markInitializationFailed(errorMsg)
      onDone(errorMsg)
      return
    }
    model.markInitialized()
    onDone("")
  }

  @OptIn(ExperimentalApi::class) // opt-in experimental flags
  override fun resetConversation(
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    initialMessages: List<Message>,
  ) {
    try {
      Log.d(TAG, "Resetting conversation for model '${model.name}'")

      val instance = model.instance as LlmModelInstance? ?: return
      instance.conversation.close()

      val engine = instance.engine
      val topK = model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
      val topP = model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
      val temperature =
        model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
      val shouldEnableImage = supportImage
      val shouldEnableAudio = supportAudio
      Log.d(TAG, "Enable image: $shouldEnableImage, enable audio: $shouldEnableAudio")

      val accelerator =
        model.getStringConfigValue(
          key = ConfigKeys.ACCELERATOR,
          defaultValue = Accelerator.GPU.label,
        )
      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      val newConversation =
        engine.createConversation(
          ConversationConfig(
            samplerConfig =
              if (accelerator == Accelerator.NPU.label || accelerator == Accelerator.TPU.label) {
                null
              } else {
                SamplerConfig(
                  topK = topK,
                  topP = topP.toDouble(),
                  temperature = temperature.toDouble(),
                )
              },
            systemInstruction = systemInstruction,
            tools = tools,
            initialMessages = initialMessages,
          )
        )
      ExperimentalFlags.enableConversationConstrainedDecoding = false
      instance.conversation = newConversation

      Log.d(TAG, "Resetting done")
    } catch (e: Exception) {
      Log.d(TAG, "Failed to reset conversation", e)
    }
  }

  override fun cleanUp(model: Model, onDone: () -> Unit) {
    if (model.instance == null) {
      return
    }

    val instance = model.instance as LlmModelInstance

    try {
      instance.conversation.close()
    } catch (e: Exception) {
      Log.e(TAG, "Failed to close the conversation: ${e.message}")
    }

    try {
      instance.engine.close()
    } catch (e: Exception) {
      Log.e(TAG, "Failed to close the engine: ${e.message}")
    }

    val onCleanUp = cleanUpListeners.remove(model.name)
    if (onCleanUp != null) {
      onCleanUp()
    }
    model.resetInitialization()

    onDone()
    Log.d(TAG, "Clean up done.")
  }

  override fun stopResponse(model: Model) {
    val instance = model.instance as? LlmModelInstance ?: return
    try {
      instance.conversation.cancelProcess()
    } catch (e: IllegalStateException) {
      Log.w(TAG, "Conversation is not alive, cannot cancel process", e)
    }
  }

  override fun runInference(
    model: Model,
    input: String,
    resultListener: ResultListener,
    cleanUpListener: CleanUpListener,
    onError: (message: String) -> Unit,
    images: List<Bitmap>,
    audioClips: List<ByteArray>,
    coroutineScope: CoroutineScope?,
    extraContext: Map<String, String>?,
  ) {
    val instance = model.instance as? LlmModelInstance
    if (instance == null) {
      onError("LlmModelInstance is not initialized.")
      return
    }

    // Set listener.
    if (!cleanUpListeners.containsKey(model.name)) {
      cleanUpListeners[model.name] = cleanUpListener
    }

    if (images.isNotEmpty() && !model.llmSupportImage) {
      onError("This model does not support image input.")
      return
    }
    if (audioClips.isNotEmpty() && !model.llmSupportAudio) {
      onError("This model does not support audio input.")
      return
    }

    val conversation = instance.conversation

    val contents = mutableListOf<Content>()
    for (image in images) {
      contents.add(Content.ImageBytes(image.toOptimizedImageByteArray()))
    }
    for (audioClip in audioClips) {
      contents.add(Content.AudioBytes(audioClip))
    }
    // add the text after image and audio for the accurate last token
    if (input.trim().isNotEmpty()) {
      contents.add(Content.Text(input))
    }

    // Set enable_thinking to true only if requested.
    val enableThinking = extraContext?.get("enable_thinking") == "true"
    val finalExtraContext: Map<String, Any> =
      if (enableThinking) {
        (extraContext ?: emptyMap()) + ("enable_thinking" to true)
      } else {
        extraContext ?: emptyMap()
      }

    try {
      conversation.sendMessageAsync(
        Contents.of(contents),
        object : MessageCallback {
          override fun onMessage(message: Message) {
            resultListener(message.toString(), false, message.channels[THOUGHT_CHANNEL])
          }

          override fun onDone() {
            resultListener("", true, null)
          }

          override fun onError(throwable: Throwable) {
            if (throwable is CancellationException) {
              Log.i(TAG, "The inference is cancelled.")
              resultListener("", true, null)
            } else {
              Log.e(TAG, "onError in sendMessageAsync", throwable)
              onError("Error: ${throwable.message ?: throwable.javaClass.simpleName}")
            }
          }
        },
        finalExtraContext,
      )
    } catch (t: Throwable) {
      Log.e(TAG, "Exception during sendMessageAsync", t)
      onError("Failed to process message: ${t.message ?: t.javaClass.simpleName}")
    }
  }

  private fun Bitmap.toOptimizedImageByteArray(): ByteArray {
    val maxDimension = 1024
    val scaledBitmap =
      if (width > maxDimension || height > maxDimension) {
        val scale = maxDimension.toFloat() / maxOf(width, height)
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()
        Bitmap.createScaledBitmap(this, newWidth, newHeight, true)
      } else {
        this
      }
    val stream = ByteArrayOutputStream()
    scaledBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
    return stream.toByteArray()
  }
}
