# NLP HW1

Please find the homework assignment instructions [here](https://docs.google.com/document/d/1K8s_Ecms0cIqRO1PKPFs2bfFVFfZpc1nFoEhtxRlCaM/edit?tab=t.0).

## Part 1
* Unigram accuracy: Validation Accuracy: 17.67%, Test Accuracy: 17.79%
* 5-gram accuracy: Validation Accuracy: 58.95%, Test Accuracy: 58.44%
* Free response:

Generating 100 characters for each prompt (Unigram):
Prompt 1: ␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣
Prompt 2: ␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣
Prompt 3: ␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣
Prompt 4: ␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣
Prompt 5: ␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣␣

Generating 100 characters for each prompt (5-gram):
Prompt 1: , "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the boy w
Prompt 2: , the boy who listen the boy who listen the boy who listen the boy who listen the boy who listen the
Prompt 3:  said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who listen the 
Prompt 4:  happy and said, "i wanted to the boy who listen the boy who listen the boy who listen the boy who l
Prompt 5:  a little girl named lily was a little girl named lily was a little girl named lily was a little gir

## Part 2
* RNN accuracy: Validation Accuracy: 0.5981, Test Accuracy: 0.5922
* Link to saved model: https://drive.google.com/file/d/1OAcYoUrWgOaagytS6xcEppB5IwezHo-i/view?usp=drive_link

## Part 3
* LSTM accuracy: Validation Accuracy: 0.6134, Test Accuracy: 0.6099
* Link to saved model: https://drive.google.com/file/d/1frPLd4hBqohipPonzTFDpLYW-IyI07K_/view?usp=drive_link
* Free response:

=== RNN Generated Text ===
Prompt: <BOS>"I'm not ready to go," said...
Generated: y.<EOS> he was so happy to help.<EOS>ing in the water.<EOS> and the box.<EOS>red and fish.<EOS> the book and saw the box

Prompt: <BOS>Lily and Max were best friend...
Generated: , a came out and saw the box.<EOS>.<EOS> they had a big mess.<EOS> the box.<EOS>red and fish.<EOS> the book and saw the 

Prompt: <BOS>He picked up the juice and...
Generated: eass.<EOS>.<EOS>rents and the box.<EOS>red and fish.<EOS> the book and saw the box.<EOS>.<EOS> they had a big mess.<EOS> the box

Prompt: <BOS>It was raining, so...
Generated: n and they were happy.<EOS> them a cool for her mom and dad.<EOS> the ball and the bird was a big they were 

Prompt: <BOS>The end of the story was...
Generated:  on the store to play with her to stay it was so happy to help her mom and dad.<EOS> the store to play w

=== LSTM Generated Text ===
Prompt: <BOS>"I'm not ready to go," sa...
Generated: y the boy was so happy and said, "what is the store.<EOS> the bird was so happy and said, "what is the s

Prompt: <BOS>Lily and Max were best fr...
Generated: .<EOS> the bird was so happy and said, "what is the store.<EOS> the bird was so happy and said, "what is the

Prompt: <BOS>He picked up the juice an...
Generated: y and said, "what is the store.<EOS> the bird was so happy and said, "what is the store.<EOS> the bird was s

Prompt: <BOS>It was raining, so
Generated: n and said, "what is the store.<EOS> the bird was so happy and said, "what is the store.<EOS> the bird was s

Prompt: <BOS>The end of the story was
Generated:  and said, "what is the store.<EOS> the bird was so happy and said, "what is the store.<EOS> the bird was so


### How does coherence compare between the vanilla RNN and LSTM? 
Vanilla RNN shows better coherence, RNN output have more varied vocabulary and sentence structures, it appear to attempts different narrative elements ("shore", "sun", "dog", "bird"). It shows some contextual awareness (responding to "raining" with "happy"), and finally it
contains more diverse content across different prompts. For LSTM output, it is severely repetitive(gets stuck in the loop "i was a big book and said"),it also shows classic symptoms of mode collapse. LSTM also have less lexical diversity and narrative variation and it fails to respond meaningfully to different prompt contexts. The LSTM's repetitive behavior suggests it may have overtrained or gotten trapped in a local minimum during optimization.

### Concretely, how do the neural methods compare with the n-gram models?
Neural has several advantages which include dynamic context(Can theoretically use unlimited context length), better generalization(Learn distributed representations rather than memorizing exact sequences), smoother probability distributions(No hard zeros for unseen sequences), and
adaptive context(Context window adjusts based on relevance, not fixed n-gram size)

To my surprise, I think N-gram have it's advantages as well. It has more coherent text generation as 5-gram output shows better local coherence. It is less repetitive as it avoids the mode collapse seen in neural models. It has more predictable behavior because of the transparent statistical approach. Lastly, it has computational efficiency, the much faster training and inference time.

### What is still lacking? What could help make these models better?

Current Limitations:

Something that is lacking is the limited semantic understanding, all models operate at character level without word/meaning comprehension. It also lack short-term memory, even neural models struggle with long-range dependencies in practice. It also suffer from mode collapse which neural models show repetitive generation patterns. Another limitation is the dataset size because tiny stories may be too small for neural models to reach full potential. It might also
lack of structural awareness: No understanding of grammar, narrative flow, or story structure

Some of the potential improvements for neural models include word-level tokenization, move from characters to subwords/words. An improvement could be attention mechanisms which is to add attention to better handle long sequences. We could also improve regularization which could be dropout, weight decay to prevent overfitting. If we have better optimization of different learning rates, optimizers.

General Improvements could be improve evaluation metrics and control mechanisms.