import os
import pandas as pd
import os
import nbformat

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

def startup():
    # read the readme.md file
    with open(path, "r") as f:
        readme_text = f.read()

    # parse the table of contents
    table_of_contents = parse_table_of_contents(readme_text)
    return table_of_contents, readme_text




def process_flow_controller():
    # from start to finish, examine the readme and end with a populated, fully functional report jupyter notebook that can be tweaked and then presented to the end-user as the final report.

    table_of_contents, readme_text = startup() # read the readme.md file and parse the table of contents

    # using the table of contents, create the sections of the report notebook
    generate_report_notebook(table_of_contents,readme_text)

    return


# What are the expected sections in a data science report notebook?
# The answer is:
# 1. Introduction
# 2. Table of Contents
# 3. Data
# 4. Data Cleaning
# 5. Data Exploration
# 6. Data Analysis
# 7. Conclusions
# 8. References

# We want to create a jupyter notebook that has the following pattern for it's sections:

# A header markdown cell with the project name
# A markdown cell with the table of contents
# A markdown cell for the Introduction (with the text from the readme.md file for the Introduction)
# add a code cell below this markdown cell for the user to add any code they want to the Introduction.

# A markdown cell for the Data Section (with the text from the readme.md file for the Data Section).
# This section is about explaining what kinds of data are in the dataset and how the data was collected.

def create_data_section(report_notebook):
    global readme_text
    # generate the data section of the report notebook
    # the data section is a markdown cell with the text from the readme.md file for the Data Section.
    # This section is about explaining what kinds of data are in the dataset and how the data was collected.

    # find the line that starts with "Data" and
    return

# A markdown cell for the Data Cleaning Section (with the text from the readme.md file for the Data Cleaning Section).
# This section is about explaining how the data was cleaned.
# what kinds of dirtiness was found in the data and how it was cleaned?
# were there any missing values? how were they handled?

# A markdown cell for the Data Exploration Section (with the text from the readme.md file for the Data Exploration Section).
# This section is about explaining how the data was explored.

# make a function that generates a jupyter notebook based on the table of contents in the readme.md file.
# the pattern it uses is:
# A header markdown cell with the project name
# For each section in the table of contents:
#   A markdown cell with the section name
#   A markdown cell with the text from the readme.md file for that section
#   add a code cell below this markdown cell for the user to add any code they want to the section.

# Helper Functions for ReadMe
def get_section_text(section_name, markdown_text):
    """
    get_section_text takes a section name and a string of markdown text and returns a string of text

    Parameters

    :param section_name: a string of text
    :type section_name: str
    :param markdown_text: a string of text
    :type markdown_text: str
    :return: a string of text
    :rtype: str
    """
    # get the text for a section from the readme.md file
    # how? find the line that starts with the section name, then find the next line that starts with a "#" symbol. The text between those two lines is the text for the section.

    # step 1: find the line that starts with the section name
    section_line = [line for line in markdown_text.split('\n') if line.lower().find(section_name.lower())>-1][0]

    # step 2: find the line number of that line
    # search the markdown text for the section line
    markdown_text = markdown_text[markdown_text.find(section_line):]
    # what is the next section name?
    next_section_name = [line for line in markdown_text.split('\n') if line.startswith("#")][0]
    markdown_text = markdown_text[:markdown_text.find(next_section_name)] # remove the next section name from the markdown text. Now the markdown text only contains the text for the section.
    # remove the section name from the markdown text
    markdown_text = markdown_text.replace(section_line, "")
    # remove the newlines
    markdown_text = markdown_text.replace("\n", " ")

    return markdown_text

def pandify_readme(readme_text, table_of_contents):
    # split the readme into sections by their headernames and put those into a pandas dataframe
    # the dataframe has two columns: section_name, section_text
    # section_name is a string
    # section_text is a string

    readme_df = pd.DataFrame(columns=["section_name", "section_text"])
    # take the table of contents and make a list of section names
    section_names = [section[0] for section in table_of_contents]
    # populate the dataframe with the section names
    readme_df["section_name"] = section_names
    # populate the dataframe with the section text
    readme_df["section_text"] = readme_df["section_name"].apply(lambda x: get_section_text(x, readme_text))

    return readme_df



def generate_report_notebook(table_of_contents,readme_text):
    # global readme_text
    # generate a jupyter notebook based on the table of contents in the readme.md file.
    # the pattern it uses is:
    # A header markdown cell with the project name
    # For each section in the table of contents:
    #   A markdown cell with the section name
    #   A markdown cell with the text from the readme.md file for that section
    #   add a code cell below this markdown cell for the user to add any code they want to the section.

    # create a jupyter notebook object
    report_notebook = nbformat.v4.new_notebook()

    # create a markdown cell with the project name
    project_name = "Project Name"
    project_name_cell = nbformat.v4.new_markdown_cell(project_name)
    # add the markdown cell to the report notebook
    report_notebook.cells.append(project_name_cell)

    # create a markdown cell with the table of contents
    table_of_contents_cell = nbformat.v4.new_markdown_cell("Table of Contents")
    # add the markdown cell to the report notebook
    report_notebook.cells.append(table_of_contents_cell)

    # create a markdown cell for the Introduction (with the text from the readme.md file for the Introduction)
    # add a code cell below this markdown cell for the user to add any code they want to the Introduction.
    # get the list of sections from the table of contents that are level 1 sections
    list_of_sections = [section for section in table_of_contents if section[1]==1]
    # get the section name for the Introduction
    introduction_section_name = list_of_sections[0][0]
    # create a markdown cell with the section name
    introduction_section_name_cell = nbformat.v4.new_markdown_cell(introduction_section_name)
    # add the markdown cell to the report notebook
    report_notebook.cells.append(introduction_section_name_cell)

    # for each section in the table of contents add the pattern of cells to the report notebook
    for section in table_of_contents:
        # get the section name and section level
        section_name = section[0]
        section_level = section[1]
        # create a markdown cell with the section name
        section_name_cell = nbformat.v4.new_markdown_cell(section_name)
        # add the markdown cell to the report notebook
        report_notebook.cells.append(section_name_cell)
        # create a markdown cell with the text from the readme.md file for that section
        section_text = get_section_text(section_name, readme_text)
        section_text_cell = nbformat.v4.new_markdown_cell(section_text)
        # add the markdown cell to the report notebook
        report_notebook.cells.append(section_text_cell)
        # add a code cell below this markdown cell for the user to add any code they want to the section.
        code_cell = nbformat.v4.new_code_cell("")
        # add the code cell to the report notebook
        report_notebook.cells.append(code_cell)

    # write the report notebook to a file
    nbformat.write(report_notebook, 'report.ipynb')

    return

process_flow_controller()




















#### Appendix
# # create a markdown cell with the text from the readme.md file for the Introduction
#     introduction_section_text = get_section_text(introduction_section_name,readme_text)
#     introduction_section_text_cell = nbformat.v4.new_markdown_cell(introduction_section_text)
#     # add the markdown cell to the report notebook
#     report_notebook.cells.append(introduction_section_text_cell)
#     # add a code cell below this markdown cell for the user to add any code they want to the Introduction.
#     introduction_section_code_cell = nbformat.v4.new_code_cell()
#     # add the code cell to the report notebook
#     report_notebook.cells.append(introduction_section_code_cell)
