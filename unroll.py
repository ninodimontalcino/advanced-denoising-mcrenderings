import sys
import os
import subprocess

def run_program():
    os.system("make > /dev/null")
    output = subprocess.check_output(['./main', '256', '10'])
    for l in output.decode().split("\n"):
        if "cycles" in l:
            print(l)

def unroll_file(file, unroll_factor):
    original_unroll_factor = unroll_factor
    current_factor = unroll_factor

    extension = "." + file.split(".")[-1]
    with open(file.replace(extension, "_unroll" + extension), "r") as f:
        with open(file, "w") as g:
            for l in f:
                if "$unroll" in l:
                    current_factor = int(l.split("$unroll ")[-1].split(" ")[0])
                elif "$end_unroll" in l or "$auto_unroll" in l:
                    current_factor = original_unroll_factor
                if '$i' in l:
                    for i in range(current_factor):
                        g.write(l.replace("$i", str(i)))
                else:
                    g.write(l.replace("$n", str(current_factor)))

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Missing arguments\nUsage: python unroll.py file test_values"
    assert "build" in os.getcwd().split("/")[-1], "Please run in the build folder"
    file = sys.argv[1]
    test_values = [int(c) for c in sys.argv[2].split(",")]
    for unroll_factor in test_values:
        print("############# Running for factor {} #############".format(unroll_factor))
        unroll_file(file, unroll_factor)
        run_program()
