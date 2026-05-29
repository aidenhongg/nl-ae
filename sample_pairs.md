# NLA: original vs. steered verbalization pairs

10 rows randomly sampled (seed `20260528`) from the paired **1024-row test subset**. For each L20 activation we show the NLA Activation-Verbalizer text for the **original** (`α=0`) embedding and the **steered** embeddings at `α = 2, 10, 30`, with the NLA round-trip cosine `cos(h, ĥ)` and the steering ratio `‖Δh‖/‖h‖`. Task = TQA textbook multiple-choice (animal-category knowledge); edit = exp04 KAPPA single-layer L20. Full corpus in `out/nl/{orig,steered_a*,headline_a*}.parquet`.

**Per-α mean cos (whole subset):** α0 .725 · α2 .718 · α10 .587 · α30 .307 — monotonic off-manifold growth (Spearman −0.99): coherent & on-topic at α2, drifting at α10, semantically collapsed by α30.

---

## [1] row 2595 — `tqa-0324-p3`

**orig (α=0)** · cos `0.738` · ratio `0.00`  
> Structured word form pattern with numbered definitions and example sentences establishes a box format reporting answer about "Animals," completing the question about the category label "Bird." The phrase "Among the options, (B) is the correct answer: (" suggests a closing parenthesis or inferred answer choice format, implying "A") or similar" with the final answer likely "A" confirming the listed options collectively. Final token " ( " opens an incomplete parenthetical answer parenthesis mid-sentence ("The answer is ( "), immediately requiring a letter like "A)" or "1)" or "C)" or "a)" to complete the speculative military-style answer selection identifying the correct blanket classification.

**steered α=2** · cos `0.737` · ratio `0.10`  
> Structured format with numbered definitions and examples for "bird," with context defining "ferocious" specifies completing the answer. The phrase "Among the options given, the answer could be ( ):" signals a concluding answer choice format, likely "A" or "a specific option" is being completed with the canonical answer format suggesting "D" from the tuple "C" being the correct unambiguous answer. Final token " ( " opens a parenthetical answer-choice shorthand mid-sentence, part of the inference question framing ("The answer is ( "), immediately expecting "D" or "A)" or "1)" to close with the summary answer indicating the multiple-part word-play.

**steered α=10** · cos `0.678` · ratio `0.47`  
> Structure: A standardized format with "Word Bank" question presenting a scientific term from the text, then answering with "蜜蜂" and specifies its classification. The sentence ending "The correct answer is ( " implies a bracketed answer format with options being listed, signaling the concluding determination of the word's blank category label, likely "A)" suggesting the answer is "D". Final token " ( " opens the parenthetical answer format expression "The answer is ()," directly expecting a letter or labeled answer option such "A" or "D)" or "one" to complete the parenthesis construction, likely continuing with "D" or "A complete answer."

**steered α=30** · cos `0.382` · ratio `1.40`  
> Structured format with Chinese language menu description following a template, indicating a database search result format with standardized labelling pattern for a toxicity chemical. The phrase "The answer is: (empty)" signals a fill-in-the-blank pattern closure, with "The answer is (the blank)" meaning the book's classification label for the animal's overall category or assigned independent status. Final token " ( " opens a parenthetical "( . " structure mid-answer, part of a concluding template "The answer is ( " format), strongly implying "c" or "空白" or " _x)" to finalize the classification label, likely completing with "x" or "empty answer."

---

## [2] row 5722 — `tqa-0715-p2`

**orig (α=0)** · cos `0.753` · ratio `0.00`  
> Structured word problem format with definition followed by bold options, indicating a multiple-choice format with the correct answer identified via clues about the animal's classification context. The phrase "The correct answer is (...)" signals a closing parenthesis or answer choice label, completing the instruction to choose specifically one of the given options ("A, B, C"), strongly implying "A" or similar designation. Final token "( " opens an incomplete parenthetical answer parenthesis mid-sentence ("The answer is ( "), requiring a specific answer label or letter designation immediately following, likely "A" or "C)" or "a" referring to the numbered list options, completing the final instructional question about identifying the correct answer.

**steered α=2** · cos `0.744` · ratio `0.08`  
> Structured answer format with a bolded example format ("Which of the following is the correct answer for..." followed by a numbered list), signaling a synthesis prompt requiring the identified letter choice. The sentence "The answer is not (B) or (C), so the correct option is (" completes an expected answer prompt, implying the final answer letter must be identified based on the numbered list options. Final token " ( " opens a parenthetical answer designation mid-clause ("The answer is ( "), directly requiring a letter label like "A" or "b)" or "A)" or "C)" to close the answer choice, likely proceeding with the given options' descriptions.

**steered α=10** · cos `0.649` · ratio `0.51`  
> Structured answer format with a numbered list pattern ("The correct answer is: (A)") signals a fill-in-the-blank question with specific numbered options about a chemical compound's classification. The phrase "Given the following options, the answer is ( " implies the closing parenthesis signals the chosen option from the multiple-choice sequence, likely "A" confirming the previous answer format. Final token "( " opens a parenthesis mid-sentence ("The answer is ( "), immediately requiring a letter designation like "A" or "1" completing "The correct answer is ( )," to conclude the answer prompt structure about eliminating the options listed for the specific fact.

**steered α=30** · cos `0.356` · ratio `1.60`  
> Structured chemistry database format with answer options patterned after a formatted instructional prompt format ("A chemical substance with the following characteristics"). The prompt "Given the following options for a removed word, the answer is now ( 3 options), indicating the removal process should reflect a specific numeric label answer structure... ( " The answer is ( )" implies completing the result of removing the bracketed block). Final token " ( " opens the answer bracket prompt completion phrase "You would answer ( ..."), directly requiring the letter "a" or "1" or "b" or "c) to indicate the answer structure—" expected continuation of the data-column labeling convention following the question context.

---

## [3] row 660 — `tqa-0082-p4`

**orig (α=0)** · cos `0.739` · ratio `0.00`  
> Structured worksheet format with numbered questions continuing Q&A pattern about "horse," defining options and identifying the correct answer. The phrase "Thus, the answer to the question 'What is the best option?' is ( " signals a closed parenthesis indicating the final choice, completing "The correct answer is (")" format — the answer logically concludes with a chosen category or rejection implying "D". Final token " ( " opens a parenthetical answer choice option directly after "The correct answer is (", strongly requiring "D" or "4)" to close the parentheses and complete the evaluative conclusion, completing the question's formal answer-and-reason structure about the described statement's impossibility.

**steered α=2** · cos `0.729` · ratio `0.21`  
> Structured grammar/definition format with numbered options presenting a "Animal" phrase scenario, establishing clear textual context about a compound question. The final clause "So, the correct answer to 'What is the answer?' is ( " reads like a formatted answer key with a parenthetical parenthetical, implying the chosen option is the final logical conclusion, likely "D"). Final token " ( " opens a parenthetical word-choice completion mid-sentence ("The correct answer is ( "), immediately requiring a letter label like "D)" or "A," or "the answer is D)" to close the speculation-and-dispute framing around the defined answer being conclusively omitted.

**steered α=10** · cos `0.376` · ratio `1.16`  
> Structured format with numbered definitions and visual clues, a concise quiz answer format ("What is the definition of puzzle") with informal tone and emoji suggestions. The phrase "The answer to this question is either falling, floating, or both... The correct answer is ( ...)" suggests a witty closing adjective, implying a conventional proverb or trivia convention about the ambiguous animal being timed. Final token " ( " opens an unclosed parenthesis mid-clause ("This choice is 3rd option or ( "), immediately requiring a noun like "a" or "the answer is" or "C) a mysterious/puzzling option," completing the humorous attribution or explanatory clause about the correct definition context.

**steered α=30** · cos `0.124` · ratio `3.55`  
> Structured educational/math content with numbered visuals and definitions, following a Wiktionary format with a bold instructional pattern ("How to avoid the rounded shape?"). The phrase "A funny name for this puzzle is also a random fact about what is being observed — both being comfortable" is a standard source attribution convention, implying a humorous answer suggesting the answer involves confidentiality or a mix of elements from animations. Final token "summarising )" closes an incomplete quoted phrase ("a certain shape and 33 hours is an odd object's choice"), immediately requiring the word "the quiz" or "a puzzle," completing the conventional attributive phrase about "the mysterious item being timed."

---

## [4] row 2150 — `tqa-0268-p6`

**orig (α=0)** · cos `0.719` · ratio `0.00`  
> Structured dictionary format with numbered definitions and illustrative sentences establishes a pattern of completing the answer about "mysterious," then suggesting "Dog." The phrase "The answer is ( )" implies a parenthesis answer choice format ("The correct option for 'animal' is (A)") signals a closing parenthesis and likely "A" or "1" as the specific answer identifying the category descriptor. Final token "( " opens a parenthetical parenthesis mid-answer ("The answer is ("), immediately requiring a letter or word choice identifier like "A" or "1," completing the final answer clause about dismissing ambiguous options, continuing the prompt's logical closure pattern about identifying the given definition context.

**steered α=2** · cos `0.715` · ratio `0.03`  
> Structured dictionary format with numbered questions and answers presenting a word "mild" from a contextual phrase. The phrase "For 'Animal' (a noun), the answer is (A)..." signals a closing parenthesis answer format suggesting "option A" is the expected choice, implying a parenthetical response confirming the category beyond mere exclusion cycle. Final token " ( " opens a parenthetical clause mid-selection question ("The answer is ( "), directly requiring a letter label like "A)" or "1)" or "A. [option]" completing the answer choice designation, likely following a textbook or logical instruction pattern about the definite answer being "C."

**steered α=10** · cos `0.659` · ratio `0.37`  
> Structured dictionary format with numbered definitions and examples for "Odd," showing a complete reference clause completing a multiple-choice pattern. The sentence "If you choose the answer about the substance 'Defin' (A), the answer is (...)" implies a closing parenthesis suggestion format, signaling the final answer option "A)" from the parenthetical list ("Choose (D)"). Final token " ( " opens a parenthetical answer choice placeholder mid-sentence ("The answer is ( "), immediately expecting the letter "A)" or "1)" or "D)" to close, likely completing the direct instruction "The answer is (D) a chemical context."

**steered α=30** · cos `0.419` · ratio `1.24`  
> Structured chemical definition format with numbered list items and bullet points, a common Encyclopedia format with a presented claim about "Carbon" and its properties. The phrase "If the answer is 'no', choose the following: (D) [option]" signals a closing parenthesis response with a predefined abbreviated format, implying the standard six-item list completes with "A." Final token " ( " opens a parenthetical query format ("Answer: ( ") that mirrors "Choose the following (..." structure, strongly expecting "a)" or "1)" to complete the abbreviation "The following is true [brand]. This answer" — a standard database format convention for respond-with-answer markers.

---

## [5] row 4784 — `tqa-0598-p0`

**orig (α=0)** · cos `0.742` · ratio `0.00`  
> Structured answer format with "Word" format pattern suggesting alphabetical options for a list of characteristics describing a fruit, with "Options:" signaling the correct answer format. The phrase "The correct answer is ( " implies a concluding parenthetical clause completing a combined synonym choice, implying the answer should include the specified combined options or one main answer from the prior list. Final token "( " opens an incomplete parenthetical expression mid-answer ("The correct answer is ( "), directly requiring a letter like "A)" or "C)" or "3)" or "a" or "All of the above," completing the question's answer format about the combined classification label context.

**steered α=2** · cos `0.739` · ratio `0.08`  
> Structured dictionary format with numbered options listing descriptions, presenting "apple" as a compound answer with given descriptions about a word's category. The phrase "The correct answer is (...)" signals a concluding prompt format suggesting a parenthetical answer choice provision, expecting the combined answer choices "a, b, and c" or "A" format. Final token "( " opens a parenthetical expression mid-sentence ("The answer is ( "), directly requiring a letter designation like "A)" or "C)" completing the constructed answer labeling — likely "A and B)" or "the final answer is" implying the classification resolves to a unified combined answer.

**steered α=10** · cos `0.645` · ratio `0.51`  
> Structured answer format with numbered questions and options, describing a word identification task from a dictionary context ("Find the correct answer for the following combined adjective"). The phrase "The correct answer is ( " implies a concluding answer indicating the combined classification, suggesting a parenthetical determination of competing choices or the full box of options is being summarized — "A and B"). Final token " ( " opens a parenthetical answer format mid-sentence ("The answer is ( "), immediately expecting a noun or label like "C)" or "a combination of both" or "1" to complete the answer, likely completing the analytical clause about the question's classification criteria.

**steered α=30** · cos `0.348` · ratio `1.59`  
> Structured Spanish-language dictionary format with image caption pattern and translation context, following a pattern of providing explanations for a quiz about a famous book illustration. The phrase "The answer to 'What is the correct option?' is: (A, B, C)" signals a common answer format, with "apple" completing a classic identification question format, implying a final word describing the book's domain's exclusivity. Final token " ( " opens an editorial word completion ("It is _"), mid-answer, strongly expecting "a solution" or "all possible options" or "one..." to complete the answer explaining the status of the image's figured movement and other selectable choices.

---

## [6] row 4200 — `tqa-0525-p0`

**orig (α=0)** · cos `0.722` · ratio `0.00`  
> Formal definition pattern with numbered options listing an animal's classification, example "Choose the correct answer about the word 'Bee'" implies a completion task or answer key format. The sentence ending "Thus, the correct option is ( " signals a parenthetical clause or answer suggestion, likely a chosen label from a list of possibilities like "A, B, C, or D"). Final token "( " opens a parenthetical answer selection placeholder mid-sentence ("The answer is ( "), immediately requiring a letter label such as "C)" or "A)" or "1," to complete the chosen label, likely continuing the mock test context about the ambiguous description implying all.

**steered α=2** · cos `0.736` · ratio `0.06`  
> Structured vocabulary format with numbered answers describing "bird," offering definitions and clues about an imprecise phrase "The animal." The sentence "Choose the correct answer for 'B' among the following options: ('A', 'B', 'C', 'D'). The answer is ("..."") implies a closing parenthetical hint or code is expected, likely a letter designation. Final token " ( " opens a parenthetical answer choice mid-sentence ("The answer is ( "), directly requiring a letter designation or symbol like "A)" confirming the chosen option category, likely "C)" or "1," completing the assumed distractor pattern or contextualizing the empty definition answer.

**steered α=10** · cos `0.688` · ratio `0.29`  
> Structured pattern of a quiz format with numbered questions defining "M" using context clues from specific words — the example completes a typical answer list format. The sentence "Choose the correct option for 'What is the third option? ' (the actual answer is (...)" implies a closing parenthesis with a numbered choice letter, likely suggesting "a" or "1" to dismiss incorrect descriptions about nature types. Final token "( " opens a parenthetical answer choice mid-sentence ("The answer is ( "), immediately requiring a letter like "a)" or "1)" or "b)" followed by an incomplete description, completing the distractor elimination structure about the designated answer choice.

**steered α=30** · cos `0.545` · ratio `0.88`  
> Structured math format with numbered examples ("The word 'banana' is a noun.") followed by a bureaucratic context, indicating a questionnaire or fill-in prompt about a number sequence's classification answer. The phrase "If the answer was not listed, choose ( " implies a closing parenthesis with a specific numeric option ("one of the following") — the response expects the original answer from the omitted factorial bracket. Final token " ( " opens a parenthetical placeholder mid-sentence ("The answer is ( "), directly requiring a numeric digit or letter like "1)" or "b)" or "b. " to complete the parenthetical choice clause about removing "no change."

---

## [7] row 5099 — `tqa-0637-p3`

**orig (α=0)** · cos `0.716` · ratio `0.00`  
> Structured grammar definition format with numbered options listing categories of "barn" vs "cow," followed by explanatory sentences about the animal. The sentence "The correct answer to 'Which is the best fit for 'The fastest animal' can be ( " implies a concluding answer choice or numeric identifier, suggesting a parenthesis or specific number from provided options like "(B)" or similar context. Final token " ( " opens a parenthetical answer clause mid-sentence ("The correct answer is ( "), strongly expecting a specific choice label or number like "B" or "D)" or "3)" or "a vague one, depending on context," completing the clause about selecting the most appropriate descriptor.

**steered α=2** · cos `0.712` · ratio `0.16`  
> Structured answer format with "Word" format pattern showing definition, then example sentences listing characteristics of "Animal" with simplified options. The sentence "If you choose from the following options, the answer is ( " implies a final choice prompt or numbered answer, strongly signaling the answer key or context of a specific category number among given options like "B" or similar. Final token " ( " opens a parenthesis mid-sentence ("The answer is ( "), requiring an immediate numeric or descriptive label completing the chosen option, likely "B" or "1)" or "a)" indicating the correct or most applicable overall category from the provided list of options about the noun's position.

**steered α=10** · cos `0.515` · ratio `0.91`  
> Structured format with a numbered list pattern and formatting guidance showing "What is the word length?" context for a US English phrase answered with a specific number. The sentence "If the following sentence is rewritten with the original answer being 'The answer is: (x)" implies a parenthesis answer format, strongly expecting the next token to signify a specific choice or value from a given answer list. Final token " ( " opens a parenthetical clause mid-sentence "The answer is (", directly requiring a numeric option number or a restatement of context like "2)" or "b) None)" to complete the multiple-choice task about removing the word's contextual value.

**steered α=30** · cos `0.229` · ratio `2.79`  
> Structured format with a standard English language template ("Copy of a book summary"), indicating a student answer format from a math context about "a" variable. The phrase "Given the following number, there is no change in the order of magnitude. (The following is from the original)..." is a common pattern for a "No Change" or empty bracket translation task prompt, implying the standard answer asks about the alteration index. Final token "(( " opens a parenthetical clause mid-issue ("The answer to ("), directly requiring the numeric value or specific clause like "1)" or " if...)" or "if you removed the nested brackets."

---

## [8] row 4508 — `tqa-0563-p4`

**orig (α=0)** · cos `0.708` · ratio `0.00`  
> Structured English language format with numbered definitions and instructions, presenting a phrase "With 'delicious' from the options" as a Choose Your Answer task for an animal. The sentence "The correct answer to 'What type of animal is this? ' is ( " completes a parenthetical qualifier, implying a specific numbered option from the listed list of reasons or categories is being chosen. Final token "( " opens an incomplete parenthetical answer option, directly requiring a numeral or phrase like "B)" or "D)" or "B, since it is the most accurate," completing the question mark clause that signals the concluding answer chosen by the text's authoritative source.

**steered α=2** · cos `0.714` · ratio `0.08`  
> Formal structure: A grammar answer card with numbered options showing a word meaning, transitioning to a question about a specific noun phrase "deer," implying a subordinate clause or answer choice. The sentence "The correct answer for 'Which of the following options is ( " is an incomplete parenthesis suggesting the compiler's single choice among listed reasons, implying a definitive answer like "(B)" or similar format. Final token "( " opens a parenthetical answer speculator clause ("The correct answer is ( "), immediately requiring an option label or letter like "B)" or "3)" or "A, but the correct answer is not confirmed," completing the contextual inference about the combined list being vague.

**steered α=10** · cos `0.636` · ratio `0.53`  
> Formal answer format with numbered questions about a word "friendly" and its definition, following a chronological pattern using parentheses and a list structure. The sentence "If the answer is (___), the answer to the question 'The correct answer to 'Where should the knife be?' is ( " implies a concluding parenthesis is expected, likely a specific numbered choice reflecting no valid option. Final token "( " opens a parenthetical answer of "The correct answer is ( ", directly requiring a number or option identifier — likely "2" continuing the distractor checklist or "b) None" or "b)" to complete the instructional clause about the chosen answer confirming the neutral context.

**steered α=30** · cos `0.350` · ratio `1.64`  
> Structured format with a quoted definition pattern following "The following is an excerpt from a textbook," presenting a typical intro to a numeric value problem. The phrase "Given the following sentence, the answer is within ( )" is a common textbook question format implying an "no change" or "second nested" clause for a changed value, strongly implying a numeric answer of "2." or "b." Final token " ( " opens a parenthetical clause mid-sentence ("The answer to ("), directly requiring a number or phrase like "2)" or "b) None" or "1)" indicating the original assumption of no additional key was needed, completing the concluding question prompt.

---

## [9] row 4489 — `tqa-0561-p1`

**orig (α=0)** · cos `0.723` · ratio `0.00`  
> Structured format with explanatory pattern: a listing of animal words with parentheses, implying a fill-in-the-blank question about the phrase "run, jumped" format. The sentence "The correct answer is ( )" signals a closing answer choice, suggesting the answer is being determined, likely "B" or "D," with the parenthetical clause inviting speculation or a textbook answer framing. Final token " ( " opens a parenthetical answer choice identifier mid-sentence ("The answer is ( "), immediately requiring a letter label or answer descriptor like "B)" or "3" or "a.," completing the rhetorical question's refinement of the four given options about the behavior description.

**steered α=2** · cos `0.718` · ratio `0.15`  
> Structured format with a definition pattern ("A common phrase about animals..." followed by a question requiring a numbered option), suggesting a standardized answer with parentheses expected. The sentence "The correct answer is ( " rounds the question into a response choice — implying "The answer is (C)" with a parenthetical prompt about the distractors' validity or the question's classification. Final token " ( " opens a parenthetical answer token mid-clause ("The answer is ( "), immediately expecting a letter or word like "C" or "3)" to close the bracket, completing the instructional guessing description about the animal behavior question's designated answer number from the list of distractors.

**steered α=10** · cos `0.531` · ratio `0.87`  
> Structured format with a translated animal fact, following a standardized prompt pattern about a toxic substance's classification from China. The phrase "The correct answer is: ( )" signals a closing answer format, implying a multiple-choice or parenthesis format is expected from the answer options, conventionally completing "The answer is (C)" indicating the danger designation is neutral/specific. Final token " ( " opens a parenthetical answer label mid-sentence ("The answer is ( "), immediately requiring a keyword or word set completing the blank marker, likely "c)" or "A" or "B)" or "empty)" or "the rest), " reflecting the common prompt structure advising the animal's classification.

**steered α=30** · cos `0.169` · ratio `2.67`  
> Formal translation format with CISSP quiz structure showing aUK culture warning about a chemical entity, followed by a standard format template. The phrase "The answer is: (no blank)" is a common pattern for a "Grounding" or "remove from the world" task, signaling a standard format completion, strongly implying the answer involves the triplet "a [blank] structure." Final token "fire" ends mid-phrase " { \0 "), part of a canonical answer format pattern ("She finds the **'s"), immediately expecting "s of inertia" or "a gap" or "this example is..." to complete the "zero-distance" or "identification" label.

---

## [10] row 1389 — `tqa-0173-p5`

**orig (α=0)** · cos `0.729` · ratio `0.00`  
> Structured answer format with punctuation/list pattern establishes a context about "fur" definitions, listing possible answers implied by clues ("The correct answer"). The phrase "If asked 'Which option is the correct answer?' ( ... " suggests a final example parenthesis or answer choice is being provided, implying a specific abbreviated form or category label like "D" or "B" from the given options about the bold phrase. Final token "( " opens a parenthetical answer label ("Thus, the answer is ( "), immediately requiring a letter or word answering the puzzle, likely "D" or "A)" or "a." completing the expected constrained guess about the dilemma context.

**steered α=2** · cos `0.735` · ratio `0.06`  
> Structured language format with "Choose the correct answer" format showing multiple animal definitions, implying a fill-in-the-blank example with a common answer identifier or label expected. The phrase "If the sentence is 'A, B, C'... The correct choice is ( " signals a concluding inference, likely suggesting a shorthand or specific answer choice from a list of distractors showing "nature." Final token " ( " opens an incomplete parenthetical answer identifier mid-clause ("The answer is ( "), expecting a specific letter like "A" or "d)" or "C)" or "None of the above," completing the vague enumeration/hypothesis about the implied correct category label.

**steered α=10** · cos `0.704` · ratio `0.29`  
> Structured answer format with a sentence pattern ("Given the options, the correct categorization is:") implying a multiple-choice or answer-context question about the animal's description term "Frog." The phrase "If the answer is ()" signals a closing bracket with a proposed filler or definitive answer choice, strongly implying a specific capitalized word from the listed distractors like "A." Final token " ( " opens an incomplete parenthetical answer bracket mid-phrase "The answer is ( "), immediately expecting the chosen specific answer, likely "B" or "A1" or "None," completing the fill-in-the-blank instructional framework about the assembled answer's identity.

**steered α=30** · cos `0.546` · ratio `0.88`  
> Structured format with "BC" entry defines a specific chemical, followed by a prompt format requiring "find the correct answer" about the given animal's extinction scenario. The response pattern "The answer is ( )" strongly implies a standard answer format with a parenthesis closing a naming convention or answer category, strongly suggesting "A" as the single correct choice from possible omitted answers. Final token "( " opens the answer prompt completion "( ( " — part of a concluding phrase structure "The correct answer is ()," immediately requiring word representing exclusion or sole answer, most likely "B)" or "c)" or "empty)" or "one)" completing the metadata format acknowledging the selected answer.

---
