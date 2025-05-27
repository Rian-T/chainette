"""Simple debugging utilities for Chainette."""

from typing import List, Optional, Any
from pydantic import BaseModel
import json

class ChainDebugger:
    """Simple debugging utility for Chain execution."""
    
    @staticmethod
    def print_debug(label: str, value: Any, truncate: int = 500) -> None:
        """Print a labeled debug message with optional truncation."""
        if isinstance(value, BaseModel):
            try:
                value_str = json.dumps(value.model_dump(), indent=2)
                if len(value_str) > truncate:
                    value_str = value_str[:truncate] + "..."
                print(f"DEBUG - {label}:\n{value_str}")
            except:
                print(f"DEBUG - {label}: {str(value)}")
        elif isinstance(value, str):
            if len(value) > truncate:
                print(f"DEBUG - {label}: {value[:truncate]}...")
            else:
                print(f"DEBUG - {label}: {value}")
        else:
            print(f"DEBUG - {label}: {value}")
    
    @staticmethod
    def print_step(step_name: str, input_model: type, output_model: type, 
                   inputs: List[BaseModel], outputs: List[BaseModel], 
                   prompt: Optional[str] = None) -> None:
        """Print debugging information for a step."""
        separator = "-" * 40
        print(f"\n{separator}\nDEBUG - Step: {step_name}\n{separator}")
        print(f"DEBUG - Input Model: {input_model.__name__}")
        print(f"DEBUG - Output Model: {output_model.__name__}")
        print(f"DEBUG - Input Count: {len(inputs)}")
        
        if inputs and prompt:
            print(f"DEBUG - Sample Prompt:\n{prompt[:500]}..." if len(prompt) > 500 else prompt)
        
        if inputs:
            ChainDebugger.print_debug("Sample Input", inputs[0])
        
        print(f"DEBUG - Output Count: {len(outputs)}")
        if outputs:
            ChainDebugger.print_debug("Sample Output", outputs[0])
