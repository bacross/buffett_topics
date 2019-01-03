### This repo looks at various topic modelling algorithms in order to extract info out of Warren Buffett's Investor Letters

## Getting started

After cloning or downloading this repo, navigate to the repo folder and create the conda environment using the
requirements.txt file:

~~~
conda env create --file requirements.txt
~~~

## Data
Buffett's Letter's can be found here: http://www.berkshirehathaway.com/letters/letters.html. 
The letters from 1977 to 2003 are offered as html. The remainder are offered only as pdfs. 
Rather than deal with messy pdf text extraction, for now, I just hand downloaded the html letters. 
It's on my todo list to tackle the pdfs, but for now we can stick to the html available letters. 
I'll leave it to the reader to decide how to efficiently get the letters. My code assumes the html is saved locally.