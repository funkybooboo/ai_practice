import tiktoken from "tiktoken";

/**
 * Encodes a given input string into tokens using a specified tokenization model.
 *
 * @param {string} input - The input string to be tokenized.
 * @param {string} [tokenId='cl100k_base'] - The identifier for the tokenization model to use. Default is 'cl100k_base'.
 * @returns {number[]} - An array of token IDs representing the encoded input.
 */
function getTokens(input, tokenId = 'cl100k_base') {
    const encoding = tiktoken.get_encoding(tokenId);
    const tokens = encoding.encode(input);
    return tokens;
}

const inputText = 'Hello World! This is the first test of tiktoken library.';
console.log(getTokens(inputText));
