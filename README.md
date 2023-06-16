# The final project for the Reproducible Research course

## Team members:
- Hassnaa Abdelghany
- Daniel Śliwiński
- Artur Skowroński

## Project overview:
A brief description or summary of the project or directory, explaining its purpose and functionality.

## Installation instructions:
You only need to copy functions and run them or load them into your functions .r file. Using "text" or "label" as names in the data frame is totally forbidden in this context. The function will rely on these names to produce output, so you need to make sure that these names are only used for the text column and label column and not other columns.

## File structure:
To add more stopwords, go to the stp file and add the new word in a new line between two quotations.

## Running Code:
you have two options to create a text file:
- you have csv file with text in each row. This means you should use csvText(). where first argument will be your text file.
- you have seperate  .txt files in a folder. run multiTextFile() and add your file path.

Next, run validLabels() to avoid error when creating train and test sample use validLabels(). it takes one argument which is data and make sure there is at least two labels for each label in the data(one to be assigned in test set and another in train set). this avoid model to face labels for the first time when training or predicting data.

### Cleaning Code
then we shall clean text. The cleanText function is designed to perform text cleaning operations on a data frame containing text data. It takes the input data frame and modifies the text column by applying a series of cleaning steps. the function replaces all newline characters (\n) in the text column with a space, removes any numeric digits present in the text column and, remove characters such as commas, backslashes, exclamation marks, question marks, colons, semicolons, quotation marks, parentheses, hyphens, and hashtags.

remove_stopwords() and stem_articles() are there to remove stopwords and perform steming. first argument is the text column from your dataframe. stopword takes additional argument which is the stp.csv like file.

## Troubleshooting:
Make sure your text or R session is in 'UTF-8'. Stay tuned for full documentation and new package.

## Contact information:
Don't hesitate to contact the creator if you face any issues with running the code.
