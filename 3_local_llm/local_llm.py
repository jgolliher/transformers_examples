"""
Local LLM Inference with HuggingFace Transformers

This module demonstrates how to run Large Language Models (LLMs) locally
using HuggingFace's transformers library. It covers model loading, tokenization,
and text generation with proper device management and optimization.

Requirements:
    pip install torch
    pip install transformers
    pip install 'accelerate>=0.26.0'  # For efficient model loading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from time import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress transformers warnings during generation
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters.

    Attributes:
        min_new_tokens: Minimum number of tokens to generate.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness (0.0 = deterministic, higher = more random).
        top_p: Nucleus sampling parameter (0.0-1.0).
        do_sample: Whether to use sampling instead of greedy decoding.
        pad_token_id: Token ID for padding (prevents warnings).
    """

    min_new_tokens: int = 1
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    pad_token_id: Optional[int] = None


class LocalLLM:
    """A wrapper class for running LLMs locally with HuggingFace transformers.

    This class handles model loading, device management, tokenization, and
    text generation in a clean, educational interface.

    Attributes:
        model_name: Name or path of the HuggingFace model.
        device: Device to run the model on (cuda/mps/cpu).
        tokenizer: HuggingFace tokenizer for the model.
        model: The loaded causal language model.
        generation_config: Configuration for text generation.
    """

    def __init__(
        self,
        model_name: str,
        generation_config: Optional[GenerationConfig] = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize the LocalLLM with a specified model.

        Args:
            model_name: HuggingFace model identifier (e.g., "microsoft/DialoGPT-medium").
            generation_config: Configuration for text generation parameters.
            torch_dtype: PyTorch data type for model weights (float16 for efficiency).
        """
        self.model_name = model_name
        self.device = self._get_optimal_device()
        self.generation_config = generation_config or GenerationConfig()
        self.torch_dtype = torch_dtype

        logger.info(f"Initializing LocalLLM with model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        # Set pad token if not available
        self._configure_tokenizer()

        logger.info("LocalLLM initialization complete!")

    def _get_optimal_device(self) -> str:
        """Determine the best available device for model inference.

        Returns:
            str: Device string ("cuda", "mps", or "cpu").

        Note:
            Priority order: CUDA > MPS (Apple Silicon) > CPU
        """
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available - using GPU: {gpu_name}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info(
                "MPS (Apple Silicon) available - using Metal Performance Shaders"
            )
        else:
            device = "cpu"
            logger.info(
                "Using CPU - consider using a smaller model for better performance"
            )

        return device

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load the tokenizer for the specified model.

        Returns:
            AutoTokenizer: Configured tokenizer instance.
        """
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True  # Required for some custom models
        )
        logger.info("Tokenizer loaded successfully")
        return tokenizer

    def _load_model(self) -> AutoModelForCausalLM:
        """Load the causal language model with optimal settings.

        Returns:
            AutoModelForCausalLM: The loaded model ready for inference.
        """
        logger.info("Loading model... (this may take a few minutes)")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,  # Use float16 for memory efficiency
            device_map={"": self.device},  # Map entire model to single device
            trust_remote_code=True,  # Required for some custom models
            low_cpu_mem_usage=True,  # Optimize CPU memory during loading
        )

        logger.info("Model loaded successfully")
        return model

    def _configure_tokenizer(self) -> None:
        """Configure tokenizer settings for generation.

        Sets pad token if not available and updates generation config.
        """
        # Set pad token to eos token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Update generation config with pad token ID
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        custom_config: Optional[GenerationConfig] = None,
    ) -> Tuple[str, float]:
        """Generate a response from a list of chat messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Example: [{"role": "user", "content": "Hello!"}]
            custom_config: Optional custom generation configuration.

        Returns:
            Tuple[str, float]: Generated response text and generation time in seconds.
        """
        start_time = time()

        # Use custom config if provided, otherwise use default
        config = custom_config or self.generation_config

        # Apply chat template to format messages
        prompt = self._format_chat_prompt(messages)

        # Tokenize input
        inputs = self._tokenize_input(prompt)

        # Generate response
        response_text = self._generate_text(inputs, config)

        generation_time = time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")

        return response_text, generation_time

    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using the model's chat template.

        Args:
            messages: List of message dictionaries.

        Returns:
            str: Formatted prompt string.
        """
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template not available: {e}")
            # Fallback to simple concatenation
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant: "

        return prompt

    def _tokenize_input(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Tokenize input prompt and move to device.

        Args:
            prompt: Input text prompt.

        Returns:
            Dict[str, torch.Tensor]: Tokenized inputs on the correct device.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=2048,  # Reasonable default for most models
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def _generate_text(
        self, inputs: Dict[str, torch.Tensor], config: GenerationConfig
    ) -> str:
        """Generate text using the model with specified configuration.

        Args:
            inputs: Tokenized input tensors.
            config: Generation configuration parameters.

        Returns:
            str: Generated response text.
        """
        with torch.no_grad():  # Disable gradient computation for inference
            # Generate tokens
            generated_tokens = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                min_new_tokens=config.min_new_tokens,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=config.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )[
                0
            ]  # Get first (and only) sequence

        # Extract only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        new_tokens = generated_tokens[input_length:]

        # Decode to text
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response_text.strip()

    def chat_interface(self) -> None:
        """Interactive chat interface for testing the model.

        Provides a simple command-line interface for chatting with the model.
        Type 'quit' or 'exit' to end the conversation.
        """
        import textwrap
        import shutil

        # Get terminal width for proper text wrapping
        terminal_width = shutil.get_terminal_size().columns
        wrap_width = min(80, terminal_width - 4)  # Leave some margin

        print(f"\n🤖 Chat with {self.model_name}")
        print("Type 'quit' or 'exit' to end the conversation")
        print("─" * wrap_width)
        print()

        conversation_history = []

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Add user message to conversation
            conversation_history.append({"role": "user", "content": user_input})

            try:
                # Show thinking indicator
                print("🤔 Thinking...", end="", flush=True)

                # Generate response
                response, gen_time = self.generate_response(conversation_history)

                # Clear thinking indicator
                print("\r" + " " * 15 + "\r", end="")

                # Add assistant response to conversation
                conversation_history.append({"role": "assistant", "content": response})

                # Format and print the response with proper wrapping
                self._print_formatted_response(response, gen_time, wrap_width)

            except Exception as e:
                # Clear thinking indicator
                print("\r" + " " * 15 + "\r", end="")
                logger.error(f"Error generating response: {e}")
                print("❌ Sorry, I encountered an error. Please try again.\n")

    def _print_formatted_response(
        self, response: str, gen_time: float, wrap_width: int
    ) -> None:
        """Print the LLM response with proper formatting and wrapping.

        Args:
            response: The generated response text.
            gen_time: Time taken to generate the response.
            wrap_width: Maximum width for text wrapping.
        """
        import textwrap

        # Clean up the response
        response = response.strip()

        # Print assistant label
        print("🤖 Assistant:")

        # Handle multi-line responses (like lists, code blocks, etc.)
        if "\n" in response:
            lines = response.split("\n")
            for line in lines:
                if line.strip():  # Skip empty lines initially
                    wrapped_lines = textwrap.wrap(
                        line,
                        width=wrap_width,
                        initial_indent="   ",  # 3 spaces for indentation
                        subsequent_indent="   ",
                    )
                    for wrapped_line in wrapped_lines:
                        print(wrapped_line)
                else:
                    print()  # Preserve intentional blank lines
        else:
            # Single paragraph response
            wrapped_lines = textwrap.wrap(
                response,
                width=wrap_width,
                initial_indent="   ",  # 3 spaces for indentation
                subsequent_indent="   ",
            )
            for wrapped_line in wrapped_lines:
                print(wrapped_line)

        # Print timing info and separator
        print(f"\n   ⏱️  Generated in {gen_time:.2f}s")
        print("─" * wrap_width)
        print()


def main() -> None:
    """Main function demonstrating LocalLLM usage."""

    # Initialize the LLM with a small, efficient model
    model_name = "h2oai/h2o-danube3-500m-chat"

    # Create custom generation config for our classification task
    classification_config = GenerationConfig(
        min_new_tokens=2,
        max_new_tokens=256,
        temperature=0.1,  # Low temperature for more deterministic classification
        do_sample=False,  # Use greedy decoding for consistency
    )

    # Initialize the LLM
    llm = LocalLLM(model_name=model_name, generation_config=classification_config)

    # Optional: Start interactive chat
    user_choice = (
        input("\n💬 Would you like to start an interactive chat? (y/n): ")
        .strip()
        .lower()
    )
    if user_choice in ["y", "yes"]:
        llm.chat_interface()


if __name__ == "__main__":
    main()
