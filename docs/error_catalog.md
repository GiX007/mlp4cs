# Error Catalog

Recurring error types observed in the MLP4CS experiments. Each section is the same paragraph used in Chapter 7.5 of the report, with a short pointer to 
one representative example in the error analysis files.

## Intro
Across the three experiments the same kinds of mistakes keep appearing in different configurations and at different rates. Each one comes from a specific 
place in the pipeline where a design tradeoff or a model behavior breaks down. This section walks through the recurring error types observed on the test 
set. Each subsection defines one error in simple words, says where it shows up across the experiments, gives the architectural reason, and quotes one 
short turn from the test set as an illustration.

---

## Multi-domain turn confusion
A multi-domain turn confusion happens when the user mentions two domains in one sentence but the pipeline picks only one of them and drops the slots of 
the other. It appears in every configuration of all three Experiments at a similar rate because it is a design choice rather than a model weakness, and 
it affects roughly three to four percent of turns on the MultiWOZ 2.2 test split. The cause is architectural. The pipeline is built to pick one domain 
per turn, so when a user combines a hotel and a restaurant request in the same utterance only one set of slots is added to the predicted state for that 
turn. The missing slots from the other domain cause the Joint Goal Accuracy to fail for that turn and for every later turn, until the missing slot 
resurfaces in a later user utterance. In dialogue PMUL0079 the user says "Okay. now could you help me find a restaurant in the expensive price range 
that is in the same area as the hotel?", the ground truth marks both hotel and restaurant as active, but the pipeline predicts restaurant only, so the 
hotel state is never refreshed for that turn.

*Find it in exp1_gpt in error_analysis_20260421_073808.txt, JGA failure block, Example 2 (PMUL0079).*

---

## Slot extraction and cumulative-state errors
Slot extraction and cumulative-state errors are cases where the dialogue state tracker records a slot value that does 
not match the ground truth. Because the belief state accumulates across all turns of a dialogue, one wrong extraction 
causes the Joint Goal Accuracy to fail for that turn and every later one. This is the dominant source of JGA failure 
in every configuration of all three Experiments. JGA stays well below half on every model, and even the best fine-tuned 
configuration in Experiment 3, ft_homo_qwen3_14b, only reaches 47.7 percent. The mistakes come in three common shapes. 
The model picks the wrong slot key, or it normalizes the value differently from the dataset (for example "5 nights" 
instead of "5"), or it misses a slot the user actually provided. The cumulative penalty turns each one of these small 
mistakes into a permanent JGA failure for the rest of the dialogue. In dialogue SNG0613 the user says "I would like to 
try saigon city." The ground truth records this as restaurant-name equal to "saigon city" because saigon city is a 
restaurant in the database. The pipeline records it as restaurant-area equal to "saigon city" instead, because the 
model sees the word city and treats it as a location. From that turn onwards the belief state is wrong and the dialogue 
cannot recover.

*Find it in exp2_homo_qwen3_14b in error_analysis_20260428_205437.txt, Hallucination block, Example 3 (SNG0613).*

---

## Hallucination
A hallucination is a response that mentions an entity, address, or phone number that does not come from the database 
results passed to the response generator. Three architectural subtypes appear across the experiments. DB-vs-prompt 
mismatch only shows up in Experiment 1, because the single LLM sees the database and the user query in one prompt and 
sometimes ignores the database, generating details from its training data instead. Empty-DB mention appears in all 
three Experiments and happens when the database lookup returned no entity but the response still names one. Booking 
confirmation also appears in all three Experiments and happens when the response confirms a reservation and emits a 
[ref] token even though the entity is not in the database results passed to the response generator. In dialogue 
PMUL4643 the user says "Please book a table for 5 at 14:30 on Wednesday at Royal Spice. I will need the reference 
number. I also need to find a place to stay." Because the user mentions both a restaurant and a hotel in one turn, the 
pipeline picks only the hotel side, queries the hotel database, and passes a single guesthouse result to the response 
generator. The response is "I have confirmed your booking at Royal Spice with reference [ref]. For accommodation, I 
found a guesthouse in the east with a moderate price range." Royal Spice is not in the database results given to the 
response generator, so the booking confirmation is made up.

*DB-vs-prompt mismatch: Find it in exp1_gpt in error_analysis_20260421_073808.txt, Hallucination block, Example 1 
(PMUL0079: DB returned "restaurant two two", the response named "curry garden" instead).*

*Empty-DB mention: Find it in exp1_gpt-nano in error_analysis_20260421_073808.txt, Hallucination block, Example 1 
(SNG02172: DB results empty, the response named "The Ashley Hotel" with a real address).*

*Booking confirmation: Find it in exp2_homo_qwen3_14b in error_analysis_20260428_205437.txt, Hallucination block, 
Example 1 (PMUL4643: DB results contain only a guesthouse, the response confirms a Royal Spice restaurant booking anyway).*

---

## Non-canonical slot placeholders

A non-canonical slot placeholder is when the dialogue state tracker stores a placeholder string like [hotel_name] as 
the value of a slot, instead of the actual entity name the user mentioned. It shows up in Experiment 1 with the 
single-LLM setup, where the same prompt that asks the model to extract slots also describes the placeholder vocabulary 
used in the response, and the model occasionally confuses the two and writes the placeholder token where it should 
write the value. The number of affected turns is small, but every affected turn loses JGA and Slot F1 because the 
placeholder string never matches the ground truth value. In dialogue SNG02172 the user says "I choose the ashley hotel. 
What is their address, please?". The ground truth slot value is hotel-name equal to "ashley hotel", but gpt-nano 
writes hotel-name equal to [hotel_name], storing the placeholder token itself in the belief state instead of the 
entity name the user provided.

*Find it in exp1_gpt-nano in error_analysis_20260421_073808.txt, Hallucination block, Example 1 (SNG02172: pred slot 
hotel-name is the literal token [hotel_name] instead of "ashley hotel").*

---

## Policy violations
A policy violation is recorded for a turn only when two conditions hold together. First, the dialogue state tracker predicted a booking intent and the 
policy layer found that at least one required booking slot is still missing. Second, the response generator's output contains a booking-confirmation signal, 
either a [ref] token in the delexicalized response or one of the words "booked", "confirmed", "reservation", "your booking", or "your table" in the 
lexicalized response. Turns where the first condition holds but the response cleanly asks for the missing slots without any of those words are not counted. 
The check is the same in both Experiments and runs after the response generator. Violations appear in every Experiment, with the highest rates in 
Experiment 2 for the strongest API models. Haiku rises from 3.6 percent in Experiment 1 to 5.1 percent in Experiment 2 for the homogeneous configuration. 
The cause is architectural and lives upstream of the metric. In Experiment 1 the single LLM has the booking policy rules in the same prompt that picks the 
intent, so it rarely predicts book_hotel or book_restaurant unless all required slots are present, and few turns ever reach the first condition. In 
Experiment 2 the DST is policy-blind because the rules live in a separate downstream layer, so the DST predicts the booking intent as soon as the user 
uses booking language. More turns reach the first condition, and the second condition fires at a similar rate across the two Experiments because response 
generators use "your booking" or "reservation" vocabulary naturally on booking-intent turns. In dialogue SNG0840 the user says "Maybe. Is either one a 4 
star hotel? If so, I'd like to book a room for 4 nights.". The Experiment 2 homo_haiku DST predicts book_hotel, hotel-bookday and hotel-bookpeople are 
missing, and the response "Perfect! I can help you book [hotel_name] for 4 nights. To complete your reservation, I just need to know..." contains the word 
"reservation", so the turn is flagged.

*Find it in api_baselines_cot_test_20260420/exp2_homo_haiku in error_analysis_20260421_073808.txt, Policy violation block, [exp2_homo_haiku] section, Example 1 (dialogue SNG0840, pred intent book_hotel, 
violations ['hotel-bookday', 'hotel-bookpeople'], response contains "reservation").*

---

## Fine-tuning hallucination amplification
A fine-tuning hallucination amplification is when the fine-tuned response generator emits real entity names where the zero-shot version of the same model 
would have emitted placeholders. It appears only in Experiment 3 and the rates jump sharply. Haiku ends Experiment 2 at 0.8 percent hallucination, while 
the fine-tuned configurations land between 14 and 46 percent: Qwen3-14B at 14 percent, Phi-4 14B at 15.5, Qwen3-8B at 22, LLaMA-3.2-3B at 27.6, and 
LLaMA-3.1-8B at 45.6. The cause is in the training data. It was generated by string-matching delexicalization on the gold MultiWOZ responses, which only 
replaced the active domain's entity names with placeholders. On multi-domain turns the non-active domain's names stayed lexicalized in the training 
response. The fine-tuned models learn this noise as signal and emit real entity names at inference, even on single-domain turns. A second visible 
signature of the same root cause is format leakage. Phi-4's chat-template token <|im_sep|> appears in some responses, and LLaMA-3.1-8B sometimes emits 
training-format role markers like "USER:" and "SYSTEM:" mid-response. In dialogue SNG0888 the user says "Four. Two nights. Beginning Saturday.", the 
database returns the entity "alpha-milton guest house" for the hotel query, but the fine-tuned Phi-4 14B response is "<|im_sep|> I have booked you at the 
hamilton lodge for 4 people for 2 nights starting Saturday. Your reference number is [ref].", a freshly invented hotel name plus the leaked chat-template 
token.

*Find it in exp3_ft_homo_phi4_14b in error_analysis_20260502_153603.txt, Hallucination block, Example 1 (dialogue SNG0888, DB returned alpha-milton guest 
house, response invented hamilton lodge, also shows <|im_sep|> token leak).*

---

## Two cross-cutting patterns
Two patterns emerge across the five mechanisms. The dominant source of failure is dialogue state tracking. Multi-domain turn confusion and slot extraction 
errors together drive Joint Goal Accuracy below half in every configuration of every Experiment, with each small extraction mistake permanently fixed in 
the cumulative belief state. The second pattern is that hallucination changes shape across the three Experiments rather than disappearing. Experiment 1 
shows DB-vs-prompt mismatch from the single LLM ignoring the database in its prompt. Experiment 2 reduces hallucination sharply for the API models because 
the placeholder rule in the response generator structurally forces it down. Experiment 3 sees the rate climb again, in some cases above forty percent, 
because the same rule cannot be enforced after fine-tuning. The non-canonical placeholder errors in Experiment 1 and the format leakage in Experiment 3 
are surface signs of the same underlying issue: the model has not learned to keep the placeholder vocabulary separate from the actual slot values.

---
