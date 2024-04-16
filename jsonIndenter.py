def add_indentation(input_file):
    with open(input_file, 'r') as file:
        content = file.read()

    output_content = ""
    indent_level = 0
    for char in content:
        if char == ',':
            output_content += char + '\n' + '\t' * indent_level
        elif char in ['{', '[']:
            output_content += char + '\n' + '\t' * (indent_level + 1)
            indent_level += 1
        elif char in ['}', ']']:
            indent_level -= 1
            output_content += '\n' + '\t' * indent_level + char
        else:
            output_content += char

    with open(input_file, 'w') as file:
        file.write(output_content)

input_file = "vgg-read.txt"
add_indentation(input_file)
