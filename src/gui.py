import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import cv2
import numpy as np
import threading

from src.texturing import annotate, texture
from src.visualizations import save_scaled, show_scaled, visualize_positions


class ParameterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parameter Configuration")
        self.root.geometry("600x800")

        # Initialize variables

        # Input images
        self.source = None
        self.elements = None
        self.mask = None

        # Processed data
        self.vector_field = None
        self.annotations = None
        self.colors = None
        self.positions = None
        self.result = None  # Added to store the result image

        self.positions_vis = None

        # Parameters
        self.density = tk.DoubleVar(value=1.0)
        self.placement_mode = tk.StringVar(value="sampling")
        self.allow_partly_in_mask = tk.BooleanVar(value=False)
        self.boundary_mask_padding = tk.IntVar(value=0)
        self.element_padding_x = tk.IntVar(value=0)
        self.element_padding_y = tk.IntVar(value=0)
        self.scale = tk.IntVar(value=4)
        # Default black color
        self.excluded_colors = [np.array([0, 0, 0, 255])]
        self.color_map_mode = tk.StringVar(value="auto")
        self.element_color_mode = tk.StringVar(value="region")
        self.hsv_shift_h = tk.IntVar(value=0)
        self.hsv_shift_s = tk.IntVar(value=0)
        self.hsv_shift_v = tk.IntVar(value=-76)
        self.max_attempts = tk.IntVar(value=1000)
        self.max_color_distance = tk.DoubleVar(value=70.0)
        self.source_file = tk.StringVar(value="")
        self.boundary_file = tk.StringVar(value="")
        self.element_sheet_file = tk.StringVar(value="")

        # Store button references for state updates
        self.display_buttons = {}
        self.save_buttons = {}

        self.create_widgets()

    def create_widgets(self):
        # Main scrollable frame
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(
            self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Density
        ttk.Label(scrollable_frame, text="Density (0-1):").grid(row=0,
                                                                column=0, sticky="w", padx=5, pady=2)
        density_scale = ttk.Scale(scrollable_frame, from_=0, to=1,
                                  variable=self.density, orient="horizontal", length=200)
        density_scale.grid(row=0, column=1, padx=5, pady=2)
        density_label = ttk.Label(scrollable_frame, text="1.0")
        density_label.grid(row=0, column=2, padx=5, pady=2)
        density_scale.configure(
            command=lambda val: density_label.configure(text=f"{float(val):.1f}"))

        # Placement Mode
        ttk.Label(scrollable_frame, text="Placement Mode:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(scrollable_frame, textvariable=self.placement_mode, values=[
                     "packed", "sampling"], state="readonly").grid(row=1, column=1, padx=5, pady=2)

        # Allow Partly in Mask
        ttk.Checkbutton(scrollable_frame, text="Allow Partly in Mask", variable=self.allow_partly_in_mask).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # Boundary Mask Padding
        ttk.Label(scrollable_frame, text="Boundary Mask Padding (-10 to 10):").grid(
            row=3, column=0, sticky="w", padx=5, pady=2)
        boundary_scale = ttk.Scale(scrollable_frame, from_=-10, to=10,
                                   variable=self.boundary_mask_padding, orient="horizontal", length=200)
        boundary_scale.grid(row=3, column=1, padx=5, pady=2)
        boundary_label = ttk.Label(scrollable_frame, text="0")
        boundary_label.grid(row=3, column=2, padx=5, pady=2)
        boundary_scale.configure(
            command=lambda val: boundary_label.configure(text=f"{int(float(val))}"))

        # Element Padding
        ttk.Label(scrollable_frame, text="Element Padding X (-5 to 5):").grid(row=4,
                                                                              column=0, sticky="w", padx=5, pady=2)
        padding_x_scale = ttk.Scale(scrollable_frame, from_=-5, to=5,
                                    variable=self.element_padding_x, orient="horizontal", length=200)
        padding_x_scale.grid(row=4, column=1, padx=5, pady=2)
        padding_x_label = ttk.Label(scrollable_frame, text="0")
        padding_x_label.grid(row=4, column=2, padx=5, pady=2)
        padding_x_scale.configure(
            command=lambda val: padding_x_label.configure(text=f"{int(float(val))}"))

        ttk.Label(scrollable_frame, text="Element Padding Y (-5 to 5):").grid(row=5,
                                                                              column=0, sticky="w", padx=5, pady=2)
        padding_y_scale = ttk.Scale(scrollable_frame, from_=-5, to=5,
                                    variable=self.element_padding_y, orient="horizontal", length=200)
        padding_y_scale.grid(row=5, column=1, padx=5, pady=2)
        padding_y_label = ttk.Label(scrollable_frame, text="0")
        padding_y_label.grid(row=5, column=2, padx=5, pady=2)
        padding_y_scale.configure(
            command=lambda val: padding_y_label.configure(text=f"{int(float(val))}"))

        # Scale
        ttk.Label(scrollable_frame, text="Scale (1-32):").grid(row=6,
                                                               column=0, sticky="w", padx=5, pady=2)
        scale_scale = ttk.Scale(scrollable_frame, from_=1, to=32,
                                variable=self.scale, orient="horizontal", length=200)
        scale_scale.grid(row=6, column=1, padx=5, pady=2)
        scale_label = ttk.Label(scrollable_frame, text="4")
        scale_label.grid(row=6, column=2, padx=5, pady=2)
        scale_scale.configure(
            command=lambda val: scale_label.configure(text=f"{int(float(val))}"))

        # Excluded Colors
        ttk.Label(scrollable_frame, text="Excluded Colors:").grid(
            row=7, column=0, sticky="w", padx=5, pady=2)
        color_frame = ttk.Frame(scrollable_frame)
        color_frame.grid(row=7, column=1, columnspan=2,
                         sticky="w", padx=5, pady=2)
        ttk.Button(color_frame, text="Add Color",
                   command=self.add_excluded_color).pack(side="left", padx=2)
        ttk.Button(color_frame, text="Clear All",
                   command=self.clear_excluded_colors).pack(side="left", padx=2)

        self.excluded_colors_listbox = tk.Listbox(scrollable_frame, height=3)
        self.excluded_colors_listbox.grid(
            row=8, column=0, columnspan=3, sticky="ew", padx=5, pady=2)

        # Add default excluded color to listbox
        self.excluded_colors_listbox.insert(tk.END, "BGRA: [0, 0, 0, 255]")

        # Color Map Mode
        ttk.Label(scrollable_frame, text="Color Map Mode:").grid(
            row=9, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(scrollable_frame, textvariable=self.color_map_mode, values=[
                     "border", "hsv", "similarity", "auto"], state="readonly").grid(row=9, column=1, padx=5, pady=2)

        # Element Color Mode
        ttk.Label(scrollable_frame, text="Element Color Mode:").grid(
            row=10, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(scrollable_frame, textvariable=self.element_color_mode, values=[
                     "region", "per-pixel"], state="readonly").grid(row=10, column=1, padx=5, pady=2)

        # HSV Shift
        ttk.Label(scrollable_frame, text="HSV Shift H (-255 to 255):").grid(row=11,
                                                                            column=0, sticky="w", padx=5, pady=2)
        hsv_h_scale = ttk.Scale(scrollable_frame, from_=-255, to=255,
                                variable=self.hsv_shift_h, orient="horizontal", length=200)
        hsv_h_scale.grid(row=11, column=1, padx=5, pady=2)
        hsv_h_label = ttk.Label(scrollable_frame, text="0")
        hsv_h_label.grid(row=11, column=2, padx=5, pady=2)
        hsv_h_scale.configure(
            command=lambda val: hsv_h_label.configure(text=f"{int(float(val))}"))

        ttk.Label(scrollable_frame, text="HSV Shift S (-255 to 255):").grid(row=12,
                                                                            column=0, sticky="w", padx=5, pady=2)
        hsv_s_scale = ttk.Scale(scrollable_frame, from_=-255, to=255,
                                variable=self.hsv_shift_s, orient="horizontal", length=200)
        hsv_s_scale.grid(row=12, column=1, padx=5, pady=2)
        hsv_s_label = ttk.Label(scrollable_frame, text="0")
        hsv_s_label.grid(row=12, column=2, padx=5, pady=2)
        hsv_s_scale.configure(
            command=lambda val: hsv_s_label.configure(text=f"{int(float(val))}"))

        ttk.Label(scrollable_frame, text="HSV Shift V (-255 to 255):").grid(row=13,
                                                                            column=0, sticky="w", padx=5, pady=2)
        hsv_v_scale = ttk.Scale(scrollable_frame, from_=-255, to=255,
                                variable=self.hsv_shift_v, orient="horizontal", length=200)
        hsv_v_scale.grid(row=13, column=1, padx=5, pady=2)
        hsv_v_label = ttk.Label(scrollable_frame, text="-76")
        hsv_v_label.grid(row=13, column=2, padx=5, pady=2)
        hsv_v_scale.configure(
            command=lambda val: hsv_v_label.configure(text=f"{int(float(val))}"))

        # Max Attempts
        ttk.Label(scrollable_frame, text="Max Attempts:").grid(
            row=14, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.max_attempts, width=10).grid(
            row=14, column=1, sticky="w", padx=5, pady=2)

        # Max Color Distance
        ttk.Label(scrollable_frame, text="Max Color Distance:").grid(
            row=15, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.max_color_distance, width=10).grid(
            row=15, column=1, sticky="w", padx=5, pady=2)

        # File paths
        ttk.Label(scrollable_frame, text="Source File:").grid(
            row=16, column=0, sticky="w", padx=5, pady=2)
        file_frame1 = ttk.Frame(scrollable_frame)
        file_frame1.grid(row=16, column=1, columnspan=2,
                         sticky="ew", padx=5, pady=2)
        ttk.Entry(file_frame1, textvariable=self.source_file,
                  width=30).pack(side="left", padx=2)
        ttk.Button(file_frame1, text="Browse", command=lambda: self.browse_file(
            self.source_file)).pack(side="left", padx=2)

        ttk.Label(scrollable_frame, text="Boundary File:").grid(
            row=17, column=0, sticky="w", padx=5, pady=2)
        file_frame2 = ttk.Frame(scrollable_frame)
        file_frame2.grid(row=17, column=1, columnspan=2,
                         sticky="ew", padx=5, pady=2)
        ttk.Entry(file_frame2, textvariable=self.boundary_file,
                  width=30).pack(side="left", padx=2)
        ttk.Button(file_frame2, text="Browse", command=lambda: self.browse_file(
            self.boundary_file)).pack(side="left", padx=2)
        ttk.Button(file_frame2, text="Clear", command=lambda: self.boundary_file.set(
            "")).pack(side="left", padx=2)

        ttk.Label(scrollable_frame, text="Element Sheet File:").grid(
            row=18, column=0, sticky="w", padx=5, pady=2)
        file_frame3 = ttk.Frame(scrollable_frame)
        file_frame3.grid(row=18, column=1, columnspan=2,
                         sticky="ew", padx=5, pady=2)
        ttk.Entry(file_frame3, textvariable=self.element_sheet_file,
                  width=30).pack(side="left", padx=2)
        ttk.Button(file_frame3, text="Browse", command=lambda: self.browse_file(
            self.element_sheet_file)).pack(side="left", padx=2)

        # Action buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.grid(row=19, column=0, columnspan=3, pady=20)

        ttk.Button(button_frame, text="Annotate",
                   command=self.annotate_action, width=15).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Texture",
                   command=self.texture_action, width=15).pack(side="left", padx=5)

        # Display buttons
        display_frame = ttk.Frame(scrollable_frame)
        display_frame.grid(row=20, column=0, columnspan=3, pady=10)

        self.display_buttons['color_map'] = ttk.Button(display_frame, text="Show Color Map",
                                                       state="disabled",
                                                       command=self.show_color_map, width=15)
        self.display_buttons['color_map'].pack(side="left", padx=3)

        self.display_buttons['positions'] = ttk.Button(display_frame, text="Show Positions",
                                                       state="disabled",
                                                       command=self.show_positions, width=15)
        self.display_buttons['positions'].pack(side="left", padx=3)

        self.display_buttons['mask'] = ttk.Button(display_frame, text="Show Mask",
                                                  state="disabled",
                                                  command=self.show_mask, width=15)
        self.display_buttons['mask'].pack(side="left", padx=3)

        self.display_buttons['result'] = ttk.Button(display_frame, text="Show Result",
                                                    state="disabled",
                                                    command=self.show_result, width=15)
        self.display_buttons['result'].pack(side="left", padx=3)

        # Save buttons
        save_frame = ttk.Frame(scrollable_frame)
        save_frame.grid(row=21, column=0, columnspan=3, pady=10)

        self.save_buttons['color_map'] = ttk.Button(save_frame, text="Save Color Map",
                                                    state="disabled",
                                                    command=self.save_color_map, width=15)
        self.save_buttons['color_map'].pack(side="left", padx=5)

        self.save_buttons['positions'] = ttk.Button(save_frame, text="Save Positions",
                                                    state="disabled",
                                                    command=self.save_positions, width=15)
        self.save_buttons['positions'].pack(side="left", padx=5)

        self.save_buttons['mask'] = ttk.Button(save_frame, text="Save Mask",
                                               state="disabled",
                                               command=self.save_mask, width=15)
        self.save_buttons['mask'].pack(side="left", padx=5)

        self.save_buttons['result'] = ttk.Button(save_frame, text="Save Result",
                                                 state="disabled",
                                                 command=self.save_result, width=15)
        self.save_buttons['result'].pack(side="left", padx=5)

        # Configure scrollable area
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure grid weights
        scrollable_frame.columnconfigure(1, weight=1)

    def update_button_states(self):
        """Update the state of all buttons based on available data"""
        # Update color map buttons
        color_map_state = "normal" if self.colors is not None else "disabled"
        self.display_buttons['color_map'].configure(state=color_map_state)
        self.save_buttons['color_map'].configure(state=color_map_state)

        # Update positions buttons
        positions_state = "normal" if self.positions_vis is not None else "disabled"
        self.display_buttons['positions'].configure(state=positions_state)
        self.save_buttons['positions'].configure(state=positions_state)

        # Update mask buttons
        mask_state = "normal" if self.mask is not None else "disabled"
        self.display_buttons['mask'].configure(state=mask_state)
        self.save_buttons['mask'].configure(state=mask_state)

        # Update result buttons
        result_state = "normal" if self.result is not None else "disabled"
        self.display_buttons['result'].configure(state=result_state)
        self.save_buttons['result'].configure(state=result_state)

    def add_excluded_color(self):
        color = colorchooser.askcolor(title="Choose excluded color")
        if color[0]:  # color[0] is RGB tuple, color[1] is hex
            # Convert RGB to BGRA (OpenCV format)
            r, g, b = [int(c) for c in color[0]]
            bgra_color = np.array([b, g, r, 255])
            self.excluded_colors.append(bgra_color)
            self.excluded_colors_listbox.insert(
                tk.END, f"BGRA: [{b}, {g}, {r}, 255]")

    def clear_excluded_colors(self):
        self.excluded_colors.clear()
        self.excluded_colors_listbox.delete(0, tk.END)

    def browse_file(self, var):
        filename = filedialog.askopenfilename(
            title="Select file",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if filename:
            var.set(filename)

    def get_parameters(self):
        """Get all current parameter values as a dictionary"""
        return {
            'density': self.density.get(),
            'placement_mode': self.placement_mode.get(),
            'allow_partly_in_mask': self.allow_partly_in_mask.get(),
            'boundary_mask_padding': self.boundary_mask_padding.get(),
            'element_padding': (self.element_padding_x.get(), self.element_padding_y.get()),
            'scale': self.scale.get(),
            'excluded_colors': np.array(self.excluded_colors) if self.excluded_colors else np.array([]).reshape(0, 4),
            'color_map_mode': self.color_map_mode.get(),
            'element_color_mode': self.element_color_mode.get(),
            'hsv_shift': (self.hsv_shift_h.get(), self.hsv_shift_s.get(), self.hsv_shift_v.get()),
            'max_attempts': self.max_attempts.get(),
            'max_color_distance': self.max_color_distance.get(),
            'source_file': self.source_file.get() if self.source_file.get() else None,
            'boundary_file': self.boundary_file.get() if self.boundary_file.get() else None,
            'element_sheet_file': self.element_sheet_file.get() if self.element_sheet_file.get() else None
        }

    def annotate_action(self):
        """Called when Annotate button is pressed"""
        params = self.get_parameters()

        # Run in separate thread to prevent GUI freezing during cv2.imshow
        def run_annotate():
            try:
                source, elements, mask, bb, vector_field, annotations = \
                    annotate(
                        params['source_file'],
                        params['element_sheet_file'],
                        params['boundary_file'],
                        params['boundary_mask_padding'],
                        params['scale']
                    )
                self.source = source
                self.elements = elements
                self.mask = mask
                self.bb = bb
                self.vector_field = vector_field
                self.annotations = annotations

                # Update button states on the main thread
                self.root.after(0, self.update_button_states)

            except Exception as e:
                # Handle errors gracefully
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Annotate function error: {str(e)}"))

        threading.Thread(target=run_annotate, daemon=True).start()

    def texture_action(self):
        """Called when Texture button is pressed"""
        params = self.get_parameters()

        # Run in separate thread to prevent GUI freezing during cv2.imshow
        def run_texture():
            try:
                cv2.destroyAllWindows()

                result, colors, positions = \
                    texture(
                        self.source,
                        self.mask,
                        self.elements,
                        self.vector_field,
                        density=params["density"],
                        placement_mode=params["placement_mode"],
                        allow_partly_in_mask=params["allow_partly_in_mask"],
                        element_padding=params["element_padding"],
                        excluded_colors=params["excluded_colors"],
                        color_map_mode=params["color_map_mode"],
                        element_color_mode=params["element_color_mode"],
                        max_color_distance=params["max_color_distance"],
                        hsv_shift=params["hsv_shift"],
                        max_attempts=params["max_attempts"],
                        result_only=False
                    )

                self.result = result  # Store the result
                self.colors = colors
                self.positions = positions
                self.positions_vis = visualize_positions(
                    self.source, positions)

                # Update button states on the main thread
                self.root.after(0, self.update_button_states)

                show_scaled("Output", result, params['scale'])
                cv2.waitKey(0)

            except Exception as e:
                # Handle errors gracefully
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"Texture function error: {str(e)}"))

        threading.Thread(target=run_texture, daemon=True).start()

    def show_color_map(self):
        """Display the color map using show_scaled"""
        if self.colors is not None:
            def display():
                try:
                    params = self.get_parameters()
                    show_scaled("Color Map", self.colors, params['scale'])
                    cv2.waitKey(0)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Error displaying color map: {str(e)}"))
            threading.Thread(target=display, daemon=True).start()
        else:
            messagebox.showwarning(
                "Warning", "No color map available. Run Texture first.")

    def show_positions(self):
        if self.positions_vis is not None:
            def display():
                try:
                    params = self.get_parameters()
                    show_scaled("Positions", self.positions_vis,
                                params['scale'])
                    cv2.waitKey(0)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Error displaying positions: {str(e)}"))
            threading.Thread(target=display, daemon=True).start()
        else:
            messagebox.showwarning(
                "Warning", "No positions available. Run Texture first.")

    def show_mask(self):
        """Display the mask using show_scaled"""
        if self.mask is not None:
            def display():
                try:
                    params = self.get_parameters()
                    mask_img = cv2.bitwise_and(
                        self.source, self.source, mask=self.mask.astype(np.uint8))
                    show_scaled("Mask", mask_img, params['scale'])
                    cv2.waitKey(0)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Error displaying mask: {str(e)}"))
            threading.Thread(target=display, daemon=True).start()
        else:
            messagebox.showwarning(
                "Warning", "No mask available. Run Annotate first.")

    def show_result(self):
        """Display the result image using show_scaled"""
        if self.result is not None:
            def display():
                try:
                    params = self.get_parameters()
                    show_scaled("Result", self.result, params['scale'])
                    cv2.waitKey(0)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror(
                        "Error", f"Error displaying result: {str(e)}"))
            threading.Thread(target=display, daemon=True).start()
        else:
            messagebox.showwarning(
                "Warning", "No result image available. Run Texture first.")

    def save_result(self):
        """Save the result image to disk"""
        if self.result is None:
            messagebox.showwarning(
                "Warning", "No result image available. Run Texture first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                save_scaled(filename, self.result, self.scale.get())
                messagebox.showinfo(
                    "Success", f"Result image saved to {filename}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to save result image: {str(e)}")

    def save_color_map(self):
        """Save the color map image to disk"""
        if self.colors is None:
            messagebox.showwarning(
                "Warning", "No color map available. Run Texture first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Color Map",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                save_scaled(filename, self.colors, self.scale.get())
                messagebox.showinfo(
                    "Success", f"Color map saved to {filename}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to save color map: {str(e)}")

    def save_positions(self):
        if self.positions_vis is None:
            messagebox.showwarning(
                "Warning", "No positions available. Run Texture first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Positions",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                save_scaled(filename, self.positions_vis, self.scale.get())
                messagebox.showinfo(
                    "Success", f"Positions saved to {filename}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Failed to save positions: {str(e)}")

    def save_mask(self):
        """Save the mask image to disk"""
        if self.mask is None:
            messagebox.showwarning(
                "Warning", "No mask available. Run Annotate first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            try:
                mask_img = cv2.bitwise_and(
                    self.source, self.source, mask=self.mask.astype(np.uint8))
                save_scaled(filename, mask_img, self.scale.get())
                messagebox.showinfo("Success", f"Mask saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save mask: {str(e)}")


def main():
    root = tk.Tk()
    app = ParameterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
