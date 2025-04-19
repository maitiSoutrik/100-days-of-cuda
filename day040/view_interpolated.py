import cv2
import numpy as np
import sys
import os

def display_images(original_path, gpu_path, cpu_path):
    """
    Loads and displays the original, GPU interpolated, and CPU interpolated images.
    """
    print(f"Loading original image: {original_path}")
    img_original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        print(f"Error: Could not load original image at {original_path}")
        return

    print(f"Loading GPU interpolated image: {gpu_path}")
    img_gpu = cv2.imread(gpu_path, cv2.IMREAD_GRAYSCALE)
    if img_gpu is None:
        print(f"Error: Could not load GPU image at {gpu_path}. Did the CUDA program run and generate it?")
        # Display only original if others fail
        cv2.imshow("Original", img_original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print(f"Loading CPU interpolated image: {cpu_path}")
    img_cpu = cv2.imread(cpu_path, cv2.IMREAD_GRAYSCALE)
    if img_cpu is None:
        print(f"Warning: Could not load CPU image at {cpu_path}. Displaying original and GPU only.")
        # Display original and GPU if CPU fails
        h, w = img_original.shape
        h_gpu, w_gpu = img_gpu.shape
        # Resize original to match GPU height for easier side-by-side viewing
        if h == 0: # Avoid division by zero
             print("Error: Original image has zero height.")
             return
        scale_factor = h_gpu / h
        img_original_resized = cv2.resize(img_original, (int(w * scale_factor), h_gpu), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack((img_original_resized, img_gpu))
        cv2.imshow("Original (Resized) vs GPU Interpolated", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- If all images loaded successfully ---
    print("Comparing GPU vs CPU...")

    # Resize original image using nearest neighbor to match the output size for comparison display
    h_out, w_out = img_gpu.shape
    img_original_resized = cv2.resize(img_original, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

    # Calculate difference images
    diff_gpu_cpu = cv2.absdiff(img_gpu, img_cpu)
    diff_gpu_orig = cv2.absdiff(img_gpu, img_original_resized)
    diff_cpu_orig = cv2.absdiff(img_cpu, img_original_resized)

    # --- Display Side-by-Side ---
    # Top row: Original (resized), GPU, CPU
    # Bottom row: Diff (GPU-Orig), Diff (CPU-Orig), Diff (GPU-CPU)
    try:
        top_row = np.hstack((img_original_resized, img_gpu, img_cpu))
        bottom_row = np.hstack((diff_gpu_orig, diff_cpu_orig, diff_gpu_cpu))

        # Make sure rows have same width before stacking vertically
        h1, w1 = top_row.shape
        h2, w2 = bottom_row.shape
        if w1 != w2:
             # Pad the narrower one or resize. Resizing is simpler.
            bottom_row = cv2.resize(bottom_row, (w1, h2), interpolation=cv2.INTER_NEAREST)
            print("Warning: Resized difference row to match top row width for display.")


        combined_display = np.vstack((top_row, bottom_row))

        window_title = "Top: Original(NN), GPU, CPU | Bottom: Diff(GPU-Orig), Diff(CPU-Orig), Diff(GPU-CPU) | Press 'q'"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL) # Make resizable
        cv2.imshow(window_title, combined_display)

        print("\nDisplaying comparison. Press 'q' in the window to quit.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        print("Window closed.")

    except Exception as e:
        print(f"Error during image stacking or display: {e}")
        print("Displaying individual images instead.")
        cv2.imshow("Original", img_original)
        cv2.imshow("GPU Interpolated", img_gpu)
        if img_cpu is not None: # Check again if CPU image was loaded before trying to show it
            cv2.imshow("CPU Interpolated", img_cpu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Default paths assume the script is run from the build/day040 directory
    # where the executable also resides and creates the output folder.
    default_output_dir = "./output_interpolated"
    default_original = "../../day014/lena_gray.png" # Relative path from build/day040

    # --- Argument Parsing Logic ---
    # This logic determines the final paths based on defaults and user arguments.

    # Initial default values
    original_path = default_original
    # Base names for constructing output paths if defaults are used
    original_base = os.path.basename(original_path)
    original_name_no_ext = os.path.splitext(original_base)[0]
    # Default upscale factor string for filenames (can be overridden later)
    upscale_suffix = "_x2.0" # Corresponds to default upscale_factor=2.0f in C++

    # Placeholder: Actual upscale factor might be passed to C++ code.
    # We infer it here based on typical C++ output naming convention.
    # A more robust solution would involve the C++ code outputting a manifest file
    # or having more predictable naming based *only* on input + mode.

    # Try to find the upscale factor from C++ default output filenames if they exist
    potential_gpu_default = os.path.join(default_output_dir, f"{original_name_no_ext}_interpolated_gpu{upscale_suffix}.png")
    potential_cpu_default = os.path.join(default_output_dir, f"{original_name_no_ext}_interpolated_cpu{upscale_suffix}.png")

    gpu_path = potential_gpu_default
    cpu_path = potential_cpu_default


    # Process command-line arguments
    arg_idx = 1
    while arg_idx < len(sys.argv):
        arg = sys.argv[arg_idx]
        if arg == "--original" and arg_idx + 1 < len(sys.argv):
            original_path = sys.argv[arg_idx + 1]
            print(f"Using Original Path from args: {original_path}")
            # Update base names for potential output inference
            original_base = os.path.basename(original_path)
            original_name_no_ext = os.path.splitext(original_base)[0]
            # Reset output paths to be re-inferred based on new input
            # This assumes the C++ code naming convention includes the factor.
            # We need to know the factor used by C++ code to guess correctly.
            # For now, we'll stick with the default factor suffix unless GPU/CPU paths are given.
            gpu_path = os.path.join(default_output_dir, f"{original_name_no_ext}_interpolated_gpu{upscale_suffix}.png")
            cpu_path = os.path.join(default_output_dir, f"{original_name_no_ext}_interpolated_cpu{upscale_suffix}.png")
            arg_idx += 2
        elif arg == "--gpu" and arg_idx + 1 < len(sys.argv):
            gpu_path = sys.argv[arg_idx + 1]
            print(f"Using GPU Path from args: {gpu_path}")
            # Try to infer CPU path by replacing _gpu with _cpu
            if "_gpu" in gpu_path:
                cpu_path = gpu_path.replace("_gpu", "_cpu")
            arg_idx += 2
        elif arg == "--cpu" and arg_idx + 1 < len(sys.argv):
            cpu_path = sys.argv[arg_idx + 1]
            print(f"Using CPU Path from args: {cpu_path}")
            arg_idx += 2
        # Allow specifying just the output dir
        elif arg == "--output_dir" and arg_idx + 1 < len(sys.argv):
             default_output_dir = sys.argv[arg_idx + 1]
             print(f"Using Output Directory from args: {default_output_dir}")
             # Reconstruct output paths based on new dir and existing base names/suffixes
             gpu_path = os.path.join(default_output_dir, os.path.basename(gpu_path)) # Keep existing filename part
             cpu_path = os.path.join(default_output_dir, os.path.basename(cpu_path)) # Keep existing filename part
             arg_idx += 2
        else:
            print(f"Warning: Unrecognized argument or missing value for '{arg}'. Ignoring.")
            arg_idx += 1


    # Final check if output files exist based on derived paths, handling camera default names
    if not os.path.isfile(gpu_path):
         print(f"Warning: Inferred/Specified GPU path does not exist: {gpu_path}")
         # Check for camera default names if the specific one wasn't found
         camera_gpu_path = os.path.join(default_output_dir, f"camera_frame_interpolated_gpu{upscale_suffix}.png")
         if os.path.isfile(camera_gpu_path):
             print(f"Found camera default GPU output: {camera_gpu_path}")
             gpu_path = camera_gpu_path
             # Infer corresponding CPU path for camera default
             cpu_path = camera_gpu_path.replace("_gpu", "_cpu")


    if not os.path.isfile(cpu_path):
        print(f"Warning: Inferred/Specified CPU path does not exist: {cpu_path}")
        # If GPU path was the camera default, we already set cpu_path accordingly above.
        # If GPU path was *not* camera default, but CPU is missing, print warning.


    # Check if the output directory exists before proceeding
    if not os.path.isdir(default_output_dir):
         print(f"Error: Output directory '{default_output_dir}' not found.")
         print("Please run the CUDA executable first to generate the output images.")
         sys.exit(1)


    print(f"\n--- Final Configuration ---")
    print(f"Using Original Image: {original_path}")
    print(f"Attempting GPU Output Image: {gpu_path}")
    print(f"Attempting CPU Output Image: {cpu_path}")
    print(f"-------------------------\n")

    # Final check before calling display
    if not os.path.isfile(original_path):
        print(f"Error: Final original image path not found: {original_path}")
        sys.exit(1)
    if not os.path.isfile(gpu_path):
        print(f"Error: Final GPU image path not found: {gpu_path}")
        print("Ensure the CUDA executable ran successfully and check paths/filenames.")
        sys.exit(1)
    # CPU path is optional, display_images handles its absence.

    display_images(original_path, gpu_path, cpu_path)
