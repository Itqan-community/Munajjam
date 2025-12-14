#!/usr/bin/env python3
"""
Simple audio player for validating corrected segments.
Displays all ayahs from Sura 067 and plays audio segments when clicked.

Note: Requires Python with tkinter support.
If you get '_tkinter' error, use miniforge3 Python:
    /Users/abdullahmosaibah/miniforge3/bin/python player.py
"""

import json
import os
import sys

# Check for tkinter availability
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError as e:
    print("Error: tkinter is not available in this Python installation.")
    print("\nTo fix this, use miniforge3 Python:")
    print("  /Users/abdullahmosaibah/miniforge3/bin/python player.py")
    print("\nOr run the provided script:")
    print("  ./run_player.sh")
    sys.exit(1)

from pydub import AudioSegment
from pydub.playback import play
import threading
import time


class AudioPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sura 067 Audio Player")
        self.root.geometry("1000x700")
        
        # Light theme colors
        self.bg_color = "#FFFFFF"  # White background
        self.text_color = "#000000"  # Black text
        self.ayah_number_color = "#0066CC"  # Blue for ayah numbers
        self.hover_color = "#E8F4F8"  # Light blue for hover
        self.border_color = "#CCCCCC"  # Light gray borders
        
        # Configure root background
        self.root.configure(bg=self.bg_color)
        
        # Paths
        self.segments_file = "data/corrected_segments/corrected_segments_067.json"
        self.audio_file = "Quran/audio/067.wav"
        
        # Data
        self.segments = []
        self.audio = None
        self.is_playing = False
        self.play_lock = threading.Lock()
        
        # Load data
        self.load_data()
        
        # Create GUI
        self.create_gui()
    
    def load_data(self):
        """Load corrected segments JSON and audio file."""
        try:
            # Load segments
            with open(self.segments_file, 'r', encoding='utf-8') as f:
                self.segments = json.load(f)
            
            # Sort by ayah_index to ensure correct order
            self.segments.sort(key=lambda x: x['ayah_index'])
            
            print(f"Loaded {len(self.segments)} ayahs from {self.segments_file}")
            
            # Load audio file
            print(f"Loading audio file: {self.audio_file}")
            self.audio = AudioSegment.from_wav(self.audio_file)
            print(f"Audio loaded: {len(self.audio) / 1000:.2f} seconds")
            
        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            messagebox.showerror("Error", f"File not found: {e}")
        except Exception as e:
            print(f"Error loading data: {e}")
            messagebox.showerror("Error", f"Error loading data: {e}")
    
    def create_gui(self):
        """Create the GUI components."""
        # Title label
        title_label = tk.Label(
            self.root,
            text="Sura 067 - Click an ayah to play audio",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.text_color,
            pady=10
        )
        title_label.pack()
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 12),
            bg=self.bg_color,
            fg="#666666",
            pady=5
        )
        self.status_label.pack()
        
        # Frame for text widget and scrollbar
        text_frame = tk.Frame(self.root, bg=self.bg_color)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar with light theme
        scrollbar = tk.Scrollbar(text_frame, bg=self.bg_color, troughcolor="#F0F0F0")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for ayahs (allows full text display with wrapping)
        self.ayah_text = tk.Text(
            text_frame,
            yscrollcommand=scrollbar.set,
            font=("Arial", 14),
            wrap=tk.WORD,
            width=80,
            height=25,
            padx=15,
            pady=15,
            bg="#FAFAFA",  # Very light gray background
            fg=self.text_color,
            selectbackground="#B3D9FF",  # Light blue selection
            selectforeground=self.text_color,
            insertbackground=self.text_color,
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.border_color,
            highlightcolor=self.ayah_number_color,
            state=tk.NORMAL  # Keep enabled for proper click detection
        )
        self.ayah_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.ayah_text.yview)
        
        # Bind click events
        self.ayah_text.bind('<Button-1>', self.on_text_click)
        self.ayah_text.bind('<Double-Button-1>', self.on_text_click)
        # Prevent text editing
        self.ayah_text.bind('<Key>', lambda e: "break")
        
        # Populate text widget
        self.populate_ayahs()
        
        # Instructions label
        instructions = tk.Label(
            self.root,
            text="Click an ayah to play its audio segment (only one can play at a time)",
            font=("Arial", 11),
            bg=self.bg_color,
            fg="#666666",
            pady=5
        )
        instructions.pack()
    
    def populate_ayahs(self):
        """Populate the text widget with ayahs."""
        self.ayah_text.delete(1.0, tk.END)
        
        for i, segment in enumerate(self.segments):
            ayah_num = segment['ayah_index'] + 1  # ayah_index is 0-based
            text = segment['corrected_text']
            
            # Get start position before inserting
            start_pos = self.ayah_text.index(tk.END)
            
            # Insert ayah with formatting
            self.ayah_text.insert(tk.END, f"Ayah {ayah_num}:\n", "ayah_number")
            self.ayah_text.insert(tk.END, f"{text}\n\n", "ayah_text")
            
            # Get end position after inserting
            end_pos = self.ayah_text.index(tk.END)
            
            # Tag the entire ayah block (number + text) for click detection
            tag_name = f"ayah_{i}"
            self.ayah_text.tag_add(tag_name, start_pos, end_pos)
            
            # Configure tag with hover effect
            self.ayah_text.tag_config(tag_name, 
                                     background="", 
                                     foreground="")
            
            # Bind click events with proper closure
            def make_click_handler(idx):
                def handler(event):
                    self.on_ayah_click(idx)
                return handler
            
            self.ayah_text.tag_bind(tag_name, "<Button-1>", make_click_handler(i))
            self.ayah_text.tag_bind(tag_name, "<Double-Button-1>", make_click_handler(i))
            
            # Add hover effect
            def make_enter_handler(idx):
                def handler(event):
                    self.ayah_text.tag_config(f"ayah_{idx}", background=self.hover_color)
                return handler
            
            def make_leave_handler(idx):
                def handler(event):
                    self.ayah_text.tag_config(f"ayah_{idx}", background="")
                return handler
            
            self.ayah_text.tag_bind(tag_name, "<Enter>", make_enter_handler(i))
            self.ayah_text.tag_bind(tag_name, "<Leave>", make_leave_handler(i))
        
        # Configure text styles with light theme
        self.ayah_text.tag_config("ayah_number", 
                                 font=("Arial", 14, "bold"), 
                                 foreground=self.ayah_number_color)
        self.ayah_text.tag_config("ayah_text", 
                                 font=("Arial", 14), 
                                 foreground=self.text_color)
        
        # Make text widget read-only but still clickable
        self.ayah_text.config(state=tk.DISABLED)
    
    def on_text_click(self, event):
        """Handle text widget click to find which ayah was clicked."""
        # Temporarily enable widget to get click position
        self.ayah_text.config(state=tk.NORMAL)
        
        # Get the character position where click occurred
        try:
            index = self.ayah_text.index(f"@{event.x},{event.y}")
            
            # Find which tags are at this position
            tags = self.ayah_text.tag_names(index)
            
            # Look for ayah tag (format: "ayah_<number>")
            clicked_ayah = None
            for tag in tags:
                if tag.startswith("ayah_") and tag != "ayah_number" and tag != "ayah_text":
                    try:
                        ayah_index = int(tag.split("_")[1])
                        if 0 <= ayah_index < len(self.segments):
                            clicked_ayah = ayah_index
                            break
                    except (ValueError, IndexError):
                        # Skip tags that don't match the pattern
                        continue
            
            # Disable widget again
            self.ayah_text.config(state=tk.DISABLED)
            
            # Trigger click if valid ayah was found
            if clicked_ayah is not None:
                self.on_ayah_click(clicked_ayah)
        except Exception as e:
            # Re-disable widget on error
            self.ayah_text.config(state=tk.DISABLED)
            print(f"Error handling click: {e}")
    
    def on_ayah_click(self, index):
        """Handle ayah click event to play audio."""
        # Check if already playing
        with self.play_lock:
            if self.is_playing:
                self.status_label.config(
                    text="Please wait for current audio to finish playing",
                    fg="orange"
                )
                self.root.after(2000, lambda: self.status_label.config(
                    text="Ready",
                    fg="gray"
                ))
                return
            
            # Mark as playing
            self.is_playing = True
        
        segment = self.segments[index]
        
        # Update status
        ayah_num = segment['ayah_index'] + 1
        self.status_label.config(
            text=f"Playing Ayah {ayah_num}...",
            fg="blue"
        )
        self.root.update()
        
        # Play audio in a separate thread to avoid blocking GUI
        thread = threading.Thread(
            target=self.play_segment,
            args=(segment,),
            daemon=True
        )
        thread.start()
    
    def play_segment(self, segment):
        """Play audio segment based on timestamps."""
        try:
            start_ms = int(segment['start'] * 1000)  # Convert seconds to milliseconds
            end_ms = int(segment['end'] * 1000)
            
            # Extract segment from audio
            audio_segment = self.audio[start_ms:end_ms]
            
            # Play the segment
            play(audio_segment)
            
            # Update status when done
            ayah_num = segment['ayah_index'] + 1
            self.root.after(0, lambda: self.status_label.config(
                text=f"Finished playing Ayah {ayah_num}",
                fg="green"
            ))
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text=f"Error: {e}",
                fg="red"
            ))
        finally:
            # Always release the lock when done
            with self.play_lock:
                self.is_playing = False


def main():
    """Main entry point."""
    root = tk.Tk()
    app = AudioPlayer(root)
    root.mainloop()


if __name__ == "__main__":
    main()

