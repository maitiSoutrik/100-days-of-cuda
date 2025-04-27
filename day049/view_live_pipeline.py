import cv2
import subprocess
import re
import os
import tempfile # For creating temporary files
import uuid     # For unique filenames

# Path to the compiled C++ executable (relative to the script location or root)
# Adjust this path based on where you run the script from and your build directory structure.
# Assumes running from the root project directory.
EXECUTABLE_PATH = "./build/day049/day049_perception_pipeline"
TEMP_IMAGE_FILENAME_BASE = "temp_cuda_frame" # Base name for temp file

def run_cuda_pipeline_on_file(image_path):
    """Executes the C++ CUDA pipeline on a given image file and parses the edge count."""
    try:
        # Ensure the executable exists
        if not os.path.exists(EXECUTABLE_PATH):
             print(f"Error: Executable not found at {EXECUTABLE_PATH}")
             print("Please build the C++ code first (e.g., using 'make day049_perception_pipeline' in the build dir).")
             return None, "Executable not found"

        # Run the C++ executable with the image path argument
        result = subprocess.run([EXECUTABLE_PATH, image_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=False) # Don't raise exception on non-zero exit

        # Decode stdout and stderr from bytes to string
        stdout_str = result.stdout.decode('utf-8', errors='ignore')
        stderr_str = result.stderr.decode('utf-8', errors='ignore')


        if result.returncode != 0:
            print(f"Error running CUDA pipeline (return code: {result.returncode}):")
            print(stderr_str)
            return None, f"Pipeline Error (Code: {result.returncode})"

        # Parse the output to find the edge count
        output_lines = stdout_str.splitlines()
        edge_count = None
        error_msg = "Edge count not found" # Default error

        # Look for the line containing "Detected Edge Pixels"
        for line in output_lines:
             print(f"[C++ Output]: {line}") # Print C++ output for debugging
             match = re.search(r"Detected Edge Pixels: (\d+)", line)
             if match:
                 edge_count = int(match.group(1))
                 error_msg = "" # Found it, clear error message
                 break # Stop searching once found

        if edge_count is None and result.returncode == 0: # Only print warning if C++ didn't report an error
             print(f"Warning: Could not parse edge count from C++ output for file {image_path}.")
             # Print full output if parsing failed but C++ exited cleanly
             print(f"--- C++ Full Output for {image_path} ---")
             print(stdout_str)
             print("-----------------------")
        elif edge_count is None and result.returncode != 0:
             # If C++ failed and we couldn't parse, the error is already printed
             pass # error_msg remains "Pipeline Error..."


        return edge_count, error_msg

    except FileNotFoundError:
        print(f"Error: Executable not found at {EXECUTABLE_PATH}")
        return None, "Executable not found"
    except Exception as e:
        print(f"An error occurred while running the pipeline on {image_path}: {e}")
        return None, f"Exception: {e}"

def main():
    print("Starting live perception pipeline viewer...")
    print(f"Using CUDA executable: {EXECUTABLE_PATH}")
    print("Press 'q' to quit.")

    # Initialize camera capture using OpenCV in Python
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0) # Green
    line_type = 2
    temp_file_path = None # Keep track of the temporary file

    try:
        while True:
            # 1. Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # 2. Save frame to a temporary file
            # Use a unique name to avoid potential conflicts if script restarts quickly
            temp_filename = f"{TEMP_IMAGE_FILENAME_BASE}_{uuid.uuid4()}.png"
            temp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
            try:
                 write_status = cv2.imwrite(temp_file_path, frame)
                 if not write_status:
                      print(f"Error: Failed to write temporary frame to {temp_file_path}")
                      edge_count = None
                      error_msg = "Temp file write failed"
                 else:
                     # 3. Run C++ pipeline on the temporary file
                     edge_count, error_msg = run_cuda_pipeline_on_file(temp_file_path)

            except Exception as e:
                 print(f"Error during file write or pipeline execution: {e}")
                 edge_count = None
                 error_msg = f"Exception: {e}"


            # 4. Display the *original captured frame* with the result/error overlaid
            if error_msg:
                 text = f"Edges: Error ({error_msg})"
                 display_color = (0, 0, 255) # Red for error
            elif edge_count is not None:
                 text = f"Edges: {edge_count}"
                 display_color = font_color
            else:
                 # Should not happen if error_msg is handled correctly, but as a fallback
                 text = "Edges: Unknown state"
                 display_color = (0, 165, 255) # Orange

            cv2.putText(frame, text, (10, 30), font, font_scale, display_color, line_type)

            # 5. Show the frame
            cv2.imshow('Live Perception Pipeline', frame)

            # 6. Clean up the temporary file *after* C++ process is done
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    # print(f"Deleted temp file: {temp_file_path}") # Uncomment for debugging
                    temp_file_path = None # Reset path after deletion
                except OSError as e:
                    print(f"Error deleting temporary file {temp_file_path}: {e}")
                    temp_file_path = None # Reset path even if deletion failed

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        # Final cleanup attempt for temp file just in case
        if temp_file_path and os.path.exists(temp_file_path):
             try:
                  os.remove(temp_file_path)
                  print(f"Cleaned up final temp file: {temp_file_path}")
             except OSError as e:
                  print(f"Error during final cleanup of {temp_file_path}: {e}")
        print("Viewer stopped.")

if __name__ == "__main__":
    main()
