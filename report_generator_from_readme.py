import os
import pandas as pd
import os

# Path to the directory containing the README file
path = 'readme.md' # same directory as this script

# read a readme.md file and make a report jupyter notebook that has the appropriate sections (from the table of contents in the readme.md file)

def parse_table_of_contents(readme_text):
    """
    parse_table_of_contents takes a string of text and returns a list of tuples

    Parameters

    :param readme_text: a string of text
    :type readme_text: str
    :return: a list of tuples
    :rtype: list
    """
    # parse the table of contents from the readme.md file
    # returns a list of tuples (section_name, section_level)
    # section_level is an integer
    # section_name is a string

    # the readme text is split with '\n' as the delimiter not actual newlines.

    # split the readme.md file into lines
    lines = readme_text.split('\n') # split on newlines

    # find the line that starts with "Table of Contents"
    table_of_contents_line = [line for line in lines if line.lower().find("table of contents")>-1][0]

    # find the line number of that line
    table_of_contents_line_number = lines.index(table_of_contents_line)

    # find the list of lines that are the table of contents
    table_of_contents_lines = lines[table_of_contents_line_number+1:]

    # remove the lines that are empty
    table_of_contents_lines = [line for line in table_of_contents_lines if line != ""]

    # remove the lines that are not section headers
    table_of_contents_lines = [line for line in table_of_contents_lines if line.startswith("#")]

    # parse the section names and section levels
    table_of_contents = []
    for line in table_of_contents_lines:
        # count the number of "#" symbols
        section_level = line.count("#")
        # remove the "#" symbols
        section_name = line.replace("#", "")
        # remove the spaces
        section_name = section_name.strip()
        # make a tuple
        section = (section_name, section_level)
        # append the tuple to the list of sections
        table_of_contents.append(section)

    return table_of_contents


# I want to test this now
# read the readme.md file
with open(path, "r") as f:
    readme_text = f.read()

# parse the table of contents
table_of_contents = parse_table_of_contents(readme_text)

# print the table of contents
for section in table_of_contents:

    # get the section name and section level
    section_name = section[0]
    section_level = section[1]

    # print the section name and section level
    print(section_name, section_level)
