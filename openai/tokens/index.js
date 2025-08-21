import tiktoken from "tiktoken";

/**
 * Tokenizes a given input string using a specified tokenization model.
 *
 * @param {string} text - The input string to be tokenized.
 * @param {string} [model='cl100k_base'] - The identifier for the tokenization model to use. Default is 'cl100k_base'.
 * @returns {number[]} - An array of token IDs representing the encoded input.
 */
function tokenizeText(text, model = 'cl100k_base') {
    const tokenizer = tiktoken.get_encoding(model);
    const tokens = tokenizer.encode(text);
    return tokens;
}

const sampleText = 'Hello World! This is the first test of tiktoken library.';
console.log(tokenizeText(sampleText));
