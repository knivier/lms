#!/usr/bin/env python3
"""
Pose log viewer: browse, filter, and visualize pose_log.jsonl.gz files.
"""
import gzip
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Matplotlib: avoid TkAgg (needs PIL/ImageTk); use Qt or fallback to system default
import matplotlib
import os
if 'MPLBACKEND' not in os.environ:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        pass  # Let matplotlib choose default

import matplotlib.pyplot as plt


class PoseLogViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Log Viewer")
        self.root.geometry("1200x800")
        
        self.entries = []
        self.current_idx = 0
        
        # Top frame: file selection
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Button(top_frame, text="Open Log File", command=self.open_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(top_frame, text="No file loaded")
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(top_frame, text="Plot Angles", command=self.plot_angles).pack(side=tk.RIGHT, padx=5)
        
        # Middle: frame browser
        browser_frame = ttk.LabelFrame(root, text="Frame Browser", padding=10)
        browser_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(browser_frame, text="◀◀", command=self.first_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(browser_frame, text="◀", command=self.prev_frame).pack(side=tk.LEFT, padx=2)
        
        self.frame_var = tk.StringVar(value="Frame: 0 / 0")
        ttk.Label(browser_frame, textvariable=self.frame_var, width=20).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(browser_frame, text="▶", command=self.next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(browser_frame, text="▶▶", command=self.last_frame).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(browser_frame, text="Jump to frame:").pack(side=tk.LEFT, padx=(20, 5))
        self.jump_entry = ttk.Entry(browser_frame, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(browser_frame, text="Go", command=self.jump_to_frame).pack(side=tk.LEFT, padx=2)
        
        # Filter
        filter_frame = ttk.LabelFrame(root, text="Filter", padding=10)
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(filter_frame, text="Elbow angle:").pack(side=tk.LEFT, padx=5)
        ttk.Label(filter_frame, text="Min:").pack(side=tk.LEFT)
        self.elbow_min = ttk.Entry(filter_frame, width=8)
        self.elbow_min.pack(side=tk.LEFT, padx=2)
        self.elbow_min.insert(0, "0")
        
        ttk.Label(filter_frame, text="Max:").pack(side=tk.LEFT, padx=(10, 0))
        self.elbow_max = ttk.Entry(filter_frame, width=8)
        self.elbow_max.pack(side=tk.LEFT, padx=2)
        self.elbow_max.insert(0, "180")
        
        ttk.Button(filter_frame, text="Apply Filter", command=self.apply_filter).pack(side=tk.LEFT, padx=10)
        ttk.Button(filter_frame, text="Clear Filter", command=self.clear_filter).pack(side=tk.LEFT, padx=5)
        
        # Main: text display
        text_frame = ttk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=("Courier", 10))
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text.yview)
        
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select Pose Log",
            filetypes=[("Gzip JSONL", "*.jsonl.gz"), ("JSONL", "*.jsonl"), ("All files", "*.*")],
            initialdir=Path(__file__).parent,
        )
        if not path:
            return
        
        self.entries = []
        self.current_idx = 0
        path_obj = Path(path)
        
        try:
            if path_obj.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.entries.append(json.loads(line))
            else:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.entries.append(json.loads(line))
            
            self.file_label.config(text=f"{path_obj.name} ({len(self.entries)} frames)")
            self.show_frame(0)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")
    
    def show_frame(self, idx):
        if not self.entries:
            self.text.delete("1.0", tk.END)
            self.text.insert("1.0", "No data loaded.")
            return
        
        idx = max(0, min(idx, len(self.entries) - 1))
        self.current_idx = idx
        entry = self.entries[idx]
        
        self.frame_var.set(f"Frame: {idx + 1} / {len(self.entries)}")
        
        # Format JSON nicely
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", json.dumps(entry, indent=2))
    
    def first_frame(self):
        self.show_frame(0)
    
    def prev_frame(self):
        self.show_frame(self.current_idx - 1)
    
    def next_frame(self):
        self.show_frame(self.current_idx + 1)
    
    def last_frame(self):
        self.show_frame(len(self.entries) - 1)
    
    def jump_to_frame(self):
        try:
            idx = int(self.jump_entry.get()) - 1
            self.show_frame(idx)
        except ValueError:
            pass
    
    def apply_filter(self):
        if not self.entries:
            return
        try:
            min_val = float(self.elbow_min.get())
            max_val = float(self.elbow_max.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid min/max values")
            return
        
        # Find frames where any elbow is in range
        matches = []
        for i, e in enumerate(self.entries):
            angles = e.get("angles", {})
            le = angles.get("left_elbow")
            re = angles.get("right_elbow")
            if (le is not None and min_val <= le <= max_val) or (re is not None and min_val <= re <= max_val):
                matches.append(i)
        
        if matches:
            self.show_frame(matches[0])
            messagebox.showinfo("Filter", f"Found {len(matches)} frames. Showing first match (frame {matches[0] + 1}).")
        else:
            messagebox.showinfo("Filter", "No frames match filter.")
    
    def clear_filter(self):
        self.show_frame(self.current_idx)
    
    def plot_angles(self):
        if not self.entries:
            messagebox.showinfo("Plot", "No data loaded.")
            return
        
        timestamps = []
        left_elbow = []
        right_elbow = []
        left_knee = []
        right_knee = []
        
        for e in self.entries:
            ts = e.get("timestamp_ms", 0) / 1000.0
            angles = e.get("angles", {})
            timestamps.append(ts)
            left_elbow.append(angles.get("left_elbow"))
            right_elbow.append(angles.get("right_elbow"))
            left_knee.append(angles.get("left_knee"))
            right_knee.append(angles.get("right_knee"))
        
        # Plot in separate matplotlib window (not embedded in tkinter)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        fig.canvas.manager.set_window_title("Angle Plot")
        
        # Elbows
        axes[0].plot(timestamps, left_elbow, label="Left elbow", marker=".", markersize=2)
        axes[0].plot(timestamps, right_elbow, label="Right elbow", marker=".", markersize=2)
        axes[0].set_ylabel("Elbow angle (deg)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Elbow Angles")
        
        # Knees
        axes[1].plot(timestamps, left_knee, label="Left knee", marker=".", markersize=2)
        axes[1].plot(timestamps, right_knee, label="Right knee", marker=".", markersize=2)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Knee angle (deg)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Knee Angles")
        
        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseLogViewer(root)
    root.mainloop()
