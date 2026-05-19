with open("D:\Pratham\Courses\CS336\assignment1-basics\data\traning_output\vocab.txt", "r", encoding="utf-8") as f:
    # Read lines and extract just the string part between b' and '
    lines = f.readlines()
    
    longest_line = ""
    longest_token = ""
    
    for line in lines:
        if ": b'" in line:
            # Extract the content inside the quotes
            token_str = line.split(": b'")[1].rsplit("'", 1)[0]
            if len(token_str) > len(longest_token):
                longest_token = token_str
                longest_line = line.strip()

    print(f"Longest token line: {longest_line}")