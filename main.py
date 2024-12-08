from src.utils import *

def main():
    """entry point of the program and is responsible for executing the optimization algorithm"""
    inputFile = in_file
    outputFileName = out_file  # Name of the output Excel file
    random.seed(42)  # Seed the random number generator with current time

    print("Starting the program...")

    # Read input data from the file
    print(f"Reading input data from {inputFile}...")
    read_initial(inputFile)

    # Create a new Excel workbook and select the active sheet
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Optimization Results"

    # Write the header row
    sheet.append(["Vertex Number", "Edge Number", "Sb", "Best Time"])
    print("Initialized Excel workbook and added headers.")

    for i in range(1):  # Run the optimization process 1 times
        print(f"Starting optimization run {i + 1}...")
        startTime = time.time()  # Record the start time
        endTime = startTime  # Initialize end time
        random_initial()  # Randomly initialize the solution
        iter_count = 0  # Reset iteration counter

        while (time.time() - startTime) < cutting_time:  # Continue until the time limit is reached
            iter_count += 1
            local_search(startTime)  # Perform local search to optimize the current solution
            endTime = time.time()  # Update the end time
        
        bestTime=transfer()[0]
        ver_num= transfer()[1]
        edge_num=transfer()[2]
        Sb=transfer()[3]
        print(f"Run {i + 1} completed. Iterations: {iter_count}. Best time: {bestTime}.")

        # Append the results to the Excel sheet
        sheet.append([ver_num, edge_num, Sb, bestTime])

    # Save the workbook to the specified file
    workbook.save(outputFileName)
    print(f"Results saved to {outputFileName}. Program completed.")
    
    return Sb

if __name__ == "__main__":
    main()