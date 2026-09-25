import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import shlex

# Define the argument structure for every command
COMMANDS = {
    "info": [],
    "types": [{"name": "substring", "type": "str", "optional": True}],
    "frames": [{"name": "first", "type": "int"}, {"name": "count", "type": "int"}, {"name": "out.wav", "type": "save_file"}],
    "unit": [{"name": "index", "type": "int"}, {"name": "out.wav", "type": "save_file"}],
    "chain": [{"name": "first_unit", "type": "int"}, {"name": "out.wav", "type": "save_file"}],
    "say": [{"name": "types (e.g. type1 type2)", "type": "str"}, {"name": "out.wav", "type": "save_file"}],
    "speak": [{"name": "score.txt", "type": "open_file"}, {"name": "out.wav", "type": "save_file"}],
    "pace": [{"name": "types", "type": "str"}, {"name": "secs/unit", "type": "float"}, {"name": "f0", "type": "float"}, {"name": "out.wav", "type": "save_file"}],
    "pipeline": [{"name": "pipeline str", "type": "str"}, {"name": "out.wav", "type": "save_file"}],
    "text": [{"name": "words", "type": "str"}, {"name": "out.wav", "type": "save_file"}],
    "file": [{"name": "in.txt", "type": "open_file"}, {"name": "out.wav", "type": "save_file"}],
    "names": [],
    "target-cost": [],
    "dur": [],
    "f0": [],
    "self-test": [],
    "bench": [{"name": "frames", "type": "int", "optional": True}]
}

class Cep6GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cep6 TTS Interface")
        self.geometry("700x600")
        self.executable = "./cep6"
        
        self.arg_vars = []  # Stores Tkinter variables for dynamic inputs
        
        self.create_widgets()
        self.update_command_preview()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Voice Directory Row ---
        ttk.Label(main_frame, text="Voice Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.var_voice = tk.StringVar()
        self.var_voice.trace_add("write", lambda *args: self.update_command_preview())
        ttk.Entry(main_frame, textvariable=self.var_voice, width=50).grid(row=0, column=1, sticky=tk.EW, pady=5, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_voice_dir).grid(row=0, column=2, pady=5)

        # --- Command Selection Row ---
        ttk.Label(main_frame, text="Command:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.var_command = tk.StringVar(value="info")
        self.cmd_dropdown = ttk.Combobox(main_frame, textvariable=self.var_command, values=list(COMMANDS.keys()), state="readonly")
        self.cmd_dropdown.grid(row=1, column=1, sticky=tk.EW, pady=5, padx=5)
        self.cmd_dropdown.bind("<<ComboboxSelected>>", self.on_command_change)

        # --- Dynamic Arguments Frame ---
        self.args_frame = ttk.LabelFrame(main_frame, text="Arguments", padding="10")
        self.args_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=10)
        self.args_frame.columnconfigure(1, weight=1)
        self.build_dynamic_args("info") # Build initial args

        # --- Command Preview ---
        ttk.Label(main_frame, text="Command Preview:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.var_preview = tk.StringVar()
        preview_entry = ttk.Entry(main_frame, textvariable=self.var_preview, state="readonly")
        preview_entry.grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=5)

        # --- Run Button ---
        self.btn_run = ttk.Button(main_frame, text="Run Command", command=self.run_command)
        self.btn_run.grid(row=4, column=0, columnspan=3, pady=15)

        # --- Output Console ---
        ttk.Label(main_frame, text="Console Output:").grid(row=5, column=0, sticky=tk.W)
        self.console = tk.Text(main_frame, height=10, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.console.grid(row=6, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        
        # Add scrollbar to console
        scrollbar = ttk.Scrollbar(main_frame, command=self.console.yview)
        scrollbar.grid(row=6, column=3, sticky=tk.NS, pady=5)
        self.console.configure(yscrollcommand=scrollbar.set)

        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)

    def on_command_change(self, event=None):
        cmd = self.var_command.get()
        self.build_dynamic_args(cmd)
        self.update_command_preview()

    def build_dynamic_args(self, cmd_name):
        # Clear existing widgets in args_frame
        for widget in self.args_frame.winfo_children():
            widget.destroy()
        
        self.arg_vars.clear()
        args = COMMANDS.get(cmd_name, [])

        if not args:
            ttk.Label(self.args_frame, text="No additional arguments required.").grid(row=0, column=0, pady=5)
            return

        for i, arg in enumerate(args):
            name_label = arg["name"]
            if arg.get("optional"):
                name_label += " (Optional)"
                
            ttk.Label(self.args_frame, text=name_label + ":").grid(row=i, column=0, sticky=tk.W, pady=5)
            
            var = tk.StringVar()
            var.trace_add("write", lambda *a: self.update_command_preview())
            self.arg_vars.append((arg, var))
            
            entry = ttk.Entry(self.args_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=5, padx=5)

            # Add Browse buttons for file types
            if arg["type"] == "open_file":
                ttk.Button(self.args_frame, text="Browse", 
                           command=lambda v=var: self.browse_file(v, save=False)).grid(row=i, column=2)
            elif arg["type"] == "save_file":
                ttk.Button(self.args_frame, text="Browse", 
                           command=lambda v=var: self.browse_file(v, save=True)).grid(row=i, column=2)

    def browse_voice_dir(self):
        directory = filedialog.askdirectory(title="Select Voice Directory")
        if directory:
            self.var_voice.set(directory)

    def browse_file(self, string_var, save=False):
        if save:
            filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        else:
            filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filepath:
            string_var.set(filepath)

    def get_command_list(self):
        cmd_list = [self.executable]
        
        voice_dir = self.var_voice.get().strip()
        if voice_dir:
            cmd_list.append(voice_dir)
        else:
            cmd_list.append("<voice-directory>") # Placeholder if empty

        cmd_list.append(self.var_command.get())

        for arg_config, var in self.arg_vars:
            val = var.get().strip()
            if val:
                cmd_list.append(val)
            elif not arg_config.get("optional"):
                # Insert placeholder for missing required arguments
                cmd_list.append(f"<{arg_config['name']}>")

        return cmd_list

    def update_command_preview(self):
        cmd_list = self.get_command_list()
        # shlex.join safely quotes strings with spaces for terminal preview
        try:
            preview = shlex.join(cmd_list)
        except AttributeError:  # Fallback for Python < 3.8
            preview = " ".join((f'"{x}"' if ' ' in x else x) for x in cmd_list)
        self.var_preview.set(preview)

    def run_command(self):
        if not self.var_voice.get().strip():
            messagebox.showerror("Error", "Please select a voice directory.")
            return

        cmd_list = [c for c in self.get_command_list() if not (c.startswith("<") and c.endswith(">"))]
        
        self.btn_run.config(state=tk.DISABLED)
        self.console.delete(1.0, tk.END)
        self.log_to_console(f"Executing: {shlex.join(cmd_list)}\n")
        self.log_to_console("-" * 40 + "\n")

        # Run in a separate thread so GUI doesn't freeze during generation
        thread = threading.Thread(target=self._execute_subprocess, args=(cmd_list,))
        thread.daemon = True
        thread.start()

    def _execute_subprocess(self, cmd_list):
        try:
            # shell=False is used for security and relies on the list structure
            process = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                self.log_to_console(line)
            
            process.wait()
            self.log_to_console(f"\n[Process exited with code {process.returncode}]\n")
            
        except FileNotFoundError:
            self.log_to_console("\nERROR: Could not find './cep6'. Please ensure the executable is in the same directory as this script, or update 'self.executable' in the code.\n")
        except Exception as e:
            self.log_to_console(f"\nERROR: {str(e)}\n")
        finally:
            self.btn_run.config(state=tk.NORMAL)

    def log_to_console(self, text):
        # Must be thread-safe for Tkinter
        self.console.after(0, lambda: self._insert_text(text))

    def _insert_text(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

if __name__ == "__main__":
    app = Cep6GUI()
    app.mainloop()
