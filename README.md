

## IB TRANSLATIONS INTERVIEW REVIEWER (IBIRVW)

**IBIRVW** facilitates the initial evaluation of oral interpreter tests. It processes source and target audio or video files, transcribes them into text, and evaluates the output based on a predefined set of criteria. The result is a downloadable .txt file containing individual scores for each criterion, examples of mistakes, and an overall score on a ten-point scale.

The tool utilizes OpenAI's Whisper large model for transcription and Mistral's large model for evaluation.

### Factors Affecting Evaluation Accuracy

The accuracy of evaluations depends on several factors:

**- Audio quality**: Clear recordings lead to better transcription and evaluation.

**- Extraneous speech**: Phrases like "Okay, I'll start now" or other non-interpretative comments from the interpreter may affect results.

**- Whisper performance**: Whisper’s speech recognition varies across languages, influencing the evaluation.

### Whisper Model Performance


Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62) (see [Whisper's GitHub ReadMe](https://github.com/openai/whisper/blob/main/README.md) for more details)

## License

IBIRVW is released under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Whisper's code and model weights are released under the MIT License. 

Mistral AI is used through API access.

### Support
For any issues or questions, please open an issue on this repository or contact zhanna.smelkova@gmail.com

