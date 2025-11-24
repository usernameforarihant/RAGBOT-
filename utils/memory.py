"""
Memory Module
Handles persistent conversation memory across sessions
"""

import json
import os
from typing import List, Dict
from pathlib import Path


class ConversationMemory:
    """Manages persistent conversation history."""
    
    def __init__(self, memory_file: str):
        """
        Initialize conversation memory.
        
        Args:
            memory_file: Path to JSON file for storing conversation
        """
        self.memory_file = memory_file
        
        # Ensure directory exists
        Path(self.memory_file).parent.mkdir(parents=True, exist_ok=True)
    
    def save_history(self, messages: List[Dict]) -> None:
        """
        Save conversation history to disk.
        
        Args:
            messages: List of message dictionaries
        """
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation history: {e}")
    
    def load_history(self) -> List[Dict]:
        """
        Load conversation history from disk.
        
        Returns:
            List of message dictionaries
        """
        if not os.path.exists(self.memory_file):
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            return []
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception as e:
                print(f"Error clearing conversation history: {e}")
    
    def add_message(self, role: str, content: str, sources: List[str] = None) -> None:
        """
        Add a single message to history.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            sources: Optional list of source documents
        """
        messages = self.load_history()
        
        message = {
            "role": role,
            "content": content
        }
        
        if sources:
            message["sources"] = sources
        
        messages.append(message)
        self.save_history(messages)
    
    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        """
        Get n most recent messages.
        
        Args:
            n: Number of recent messages to retrieve
            
        Returns:
            List of recent message dictionaries
        """
        messages = self.load_history()
        return messages[-n:] if messages else []