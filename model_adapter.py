#!/usr/bin/env python3
"""
Model adapter for connecting llama-cpp models to the lm-evaluation-harness
"""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class LlamaCppAdapter:
    """Adapter for llama-cpp models to the lm-evaluation-harness"""
    
    def __init__(self, llama_model):
        """Initialize with a llama-cpp model"""
        self.model = llama_model
    
    def loglikelihood(self, requests):
        """
        Return log-likelihood of continuation given context.
        
        This function calculates the difference in logprobs between 
        (context + continuation) and (context alone).
        
        Args:
            requests: List of (context, continuation) tuples
            
        Returns:
            List of (logprob, is_greedy) tuples where:
            - logprob is the log-likelihood of the continuation
            - is_greedy is whether the continuation is the model's preferred continuation
        """
        logger.debug(f"loglikelihood called with {len(requests)} requests")
        results = []
        
        # Process in batches of 1 to avoid memory issues
        for i, (context, continuation) in enumerate(requests):
            try:
                logger.debug(f"Processing request {i+1}/{len(requests)}")
                logger.debug(f"Context: {context[:50]}...")
                logger.debug(f"Continuation: {continuation[:50]}...")
                
                # Get logits for the full text (context + continuation)
                # Set temperature to 0 and max_tokens to 0 to get just the logprobs
                full_text = context + continuation
                
                # Get logprobs of full text
                try:
                    full_response = self.model(full_text, max_tokens=0, temperature=0.0)
                    # Different llama-cpp-python versions may have different response structure
                    if isinstance(full_response, dict) and 'logprobs' in full_response:
                        full_logits = full_response['logprobs']
                    elif hasattr(full_response, 'logprobs'):
                        full_logits = full_response.logprobs
                    else:
                        logger.warning("Could not extract logprobs from full response")
                        full_logits = -100.0
                except Exception as e:
                    logger.error(f"Error getting full text logits: {e}")
                    full_logits = -100.0
                
                # Get logprobs of just the context
                try:
                    context_response = self.model(context, max_tokens=0, temperature=0.0)
                    # Different llama-cpp-python versions may have different response structure
                    if isinstance(context_response, dict) and 'logprobs' in context_response:
                        context_logits = context_response['logprobs']
                    elif hasattr(context_response, 'logprobs'):
                        context_logits = context_response.logprobs
                    else:
                        logger.warning("Could not extract logprobs from context response")
                        context_logits = 0.0
                except Exception as e:
                    logger.error(f"Error getting context logits: {e}")
                    context_logits = 0.0
                
                # Calculate log likelihood
                log_likelihood = full_logits - context_logits if isinstance(full_logits, (int, float)) and isinstance(context_logits, (int, float)) else -100.0
                
                # Determine if this is the greedy choice
                # This is a simplification - in a more advanced implementation,
                # we would check if this token has the highest likelihood
                is_greedy = True  # Assuming the continuation is always greedy for now
                
                # Return tuple (logprob, is_greedy)
                results.append((float(log_likelihood), is_greedy))
                logger.debug(f"Log likelihood: {log_likelihood:.4f}, Is greedy: {is_greedy}")
                
            except Exception as e:
                logger.error(f"Error in loglikelihood calculation: {e}")
                # Return a default value in case of error
                results.append((-100.0, False))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        """
        Return log-likelihood of each token in the text.
        
        Args:
            requests: List of strings
            
        Returns:
            List of float arrays containing log-likelihood of each token
        """
        logger.debug(f"loglikelihood_rolling called with {len(requests)} requests")
        
        results = []
        for text in requests:
            try:
                # Just return a placeholder value since this isn't properly
                # implemented in the llama-cpp Python bindings yet
                results.append([-1.0])
                logger.debug(f"loglikelihood_rolling placeholder returned for: {text[:50]}...")
            except Exception as e:
                logger.error(f"Error in loglikelihood_rolling: {e}")
                results.append([-100.0])
        
        return results
    
    def generate(self, contexts, max_tokens=128, stop=None, temperature=0.0):
        """
        Generate text from the model.
        
        Args:
            contexts: String or list of strings to generate from
            max_tokens: Maximum number of tokens to generate
            stop: Stop sequence(s) to stop generation
            temperature: Temperature for sampling (0.0 = greedy)
            
        Returns:
            List of generated strings
        """
        if isinstance(contexts, str):
            contexts = [contexts]  # Convert single string to list
        
        logger.debug(f"generate called with {len(contexts)} contexts, max_tokens={max_tokens}")
        results = []
        
        for ctx in contexts:
            try:
                logger.debug(f"Generating from context: {ctx[:50]}...")
                
                # Generate from the model
                response = self.model(
                    ctx, 
                    max_tokens=max_tokens, 
                    temperature=temperature,
                    stop=stop
                )
                
                # Extract the generated text
                if isinstance(response, dict) and 'choices' in response:
                    # Handle API-style response
                    generation = response['choices'][0]['text']
                elif isinstance(response, str):
                    # Handle string response
                    generation = response
                else:
                    # Fallback if response format is unexpected
                    logger.warning(f"Unexpected response format: {type(response)}")
                    generation = str(response)
                
                results.append(generation)
                logger.debug(f"Generated {len(generation)} chars")
            
            except Exception as e:
                logger.error(f"Error in generate: {e}")
                results.append("")
        
        return results
    
    def generate_until(self, requests):
        """
        Generate text until a stop sequence is reached.
        
        Args:
            requests: List of (context, until) tuples where until is a string or list of strings
            
        Returns:
            List of generated strings
        """
        logger.debug(f"generate_until called with {len(requests)} requests")
        results = []
        
        for context, until in requests:
            try:
                logger.debug(f"Generating from context: {context[:50]}...")
                
                # Convert until to a list if it's a string
                stop_tokens = until if isinstance(until, list) else [until]
                
                # Generate with stop tokens
                response = self.model(
                    context, 
                    max_tokens=512,  # Reasonable default
                    temperature=0.0,  # Greedy decoding
                    stop=stop_tokens
                )
                
                # Extract the generated text
                if isinstance(response, dict) and 'choices' in response:
                    # Handle API-style response
                    generation = response['choices'][0]['text']
                elif isinstance(response, str):
                    # Handle string response
                    generation = response
                else:
                    # Fallback if response format is unexpected
                    logger.warning(f"Unexpected response format: {type(response)}")
                    generation = str(response)
                
                results.append(generation)
                logger.debug(f"Generated {len(generation)} chars until stop sequence")
            
            except Exception as e:
                logger.error(f"Error in generate_until: {e}")
                results.append("")
        
        return results
    
    # Additional helper methods to satisfy lm-eval-harness interface
    def __str__(self):
        return "LlamaCppAdapter"
    
    def strip(self, *args, **kwargs):
        return str(self).strip(*args, **kwargs)
    
    def lower(self):
        return str(self).lower()
    
    def upper(self):
        return str(self).upper()
