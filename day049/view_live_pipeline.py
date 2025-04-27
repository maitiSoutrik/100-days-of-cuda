import cv2
import subprocess
import re
import os

# Path to the compiled C++ executable (relative to the script location or root)
# Adjust this path based on where you run the script from and your build directory structure.
# Assumes running from the root project directory.
EXECUTABLE_PATH = "./build/day049/day049_perception_pipeline"

def run_cuda_pipeline():
    """Executes the C++ CUDA pipeline and parses the edge count."""
    try:
        # Ensure the executable exists
        if not os.path.exists(EXECUTABLE_PATH):
             print(f"Error: Executable not found at {EXECUTABLE_PATH}")
             print("Please build the C++ code first (e.g., using 'make day049_perception_pipeline' in the build dir).")
             return None, "Executable not found"

        # Run the C++ executable and capture its output (compatible with older Python 3)
        result = subprocess.run([EXECUTABLE_PATH],
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
             print(f"Warning: Could not parse edge count from C++ output.")
             # Print full output if parsing failed but C++ exited cleanly
             print("--- C++ Full Output ---")
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
        print(f"An error occurred while running the pipeline: {e}")
        return None, f"Exception: {e}"

def main():
    print("Starting live perception pipeline viewer...")
    print(f"Using CUDA executable: {EXECUTABLE_PATH}")
    print("Press 'q' to quit.")

    # Initialize camera capture using OpenCV in Python for display
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera for Python display.")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0) # Green
    line_type = 2

    frame_count = 0
    display_interval = 5 # Run pipeline every N frames to reduce overhead if needed, 1 = every frame
    last_edge_count = 0
    last_error_msg = ""

    while True:
        # Capture frame for display *in Python*
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame for display.")
            break

        frame_count += 1

        # Run the C++ pipeline periodically
        if frame_count % display_interval == 0:
             edge_count, error_msg = run_cuda_pipeline()
             if edge_count is not None:
                  last_edge_count = edge_count
                  last_error_msg = ""
             else:
                  # Keep last known count, but show error
                  last_error_msg = error_msg


        # Display the edge count on the frame
        if last_error_msg:
             text = f"Edges: Error ({last_error_msg})"
             display_color = (0, 0, 255) # Red for error
        else:
             text = f"Edges: {last_edge_count}"
             display_color = font_color

        cv2.putText(frame, text, (10, 30), font, font_scale, display_color, line_type)

        # Show the frame
        cv2.imshow('Live Perception Pipeline', frame)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Viewer stopped.")

if __name__ == "__main__":
    main()
