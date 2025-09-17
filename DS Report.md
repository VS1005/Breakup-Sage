# Data Science Report: Fine-Tuning and Evaluation of Breakup-Sage

## 1. Fine-Tuning Setup

### 1.1 Data

* **Dataset Source:**
  DailyDialog (Kaggle packaged copy) — was a high-quality, human-written multi-turn English dialogue corpus.

* **Why chosen:**
  It contained everyday conversations with utterance-level **dialogue-act** and **emotion** annotations, which made it suitable for training intent- and emotion-aware response agents.

* **Key stats (concise):**

  * **\~13,118** dialogues, **\~7.9** turns per dialogue, short utterances (avg ≈ 14–15 tokens).
  * Canonical split: **11,118 / 1,000 / 1,000** (train / val / test).
  * Since the dataset had already been processed and split, I directly used it by downloading and using the train and val (for evaluation) files during training.

* **Annotations & topics:**

  * Utterance-level **dialogue acts** and **emotion** labels were provided.
  * Topics covered Relationship, Ordinary Life, Work, Travel, etc. — relationship-related examples were well represented.

---

### 1.2 Fine-Tuning Method

* **Model Architecture:**

  * **Base Model:** LLaVA (Large Language and Vision Assistant), specifically the unsloth variant (`unsloth/llava-1.5-7b-hf-bnb-4bit`) for 4-bit quantized, memory-efficient training.
  * **Adapters:** Parameter-Efficient Fine-Tuning (PEFT) via LoRA (Low-Rank Adaptation) was used to enable fast and lightweight adaptation for the emotional recovery domain.

#### Training Pipeline

1. **Loaded Model and Processor**

   * Set `model_id` to `"unsloth/llava-1.5-7b-hf-bnb-4bit"`.
   * Loaded the processor using `AutoProcessor.from_pretrained(model_id)`.
   * Loaded the model using `AutoModelForVision2Seq.from_pretrained` with `load_in_4bit=True` and `device_map="auto"`.

2. **Prepared Model for Training**

   * Applied LoRA (Low-Rank Adaptation) using `LoraConfig` and `get_peft_model`.
   * Enabled gradient checkpointing for memory efficiency.

3. **Prepared Dataset**

   * Mounted Google Drive and set paths to training (`train.csv`) and validation (`validation.csv`) datasets.
   * Read CSVs into Pandas DataFrames.
   * Converted DataFrames to HuggingFace `Dataset` objects.

4. **Tokenized Data**

   * Defined a tokenization function to clean dialogue, actions, and emotions, obviously the model tokenizer was used for this task.
   * Tokenized both train and validation Dataset objects with truncation and a max length=1024 due to GPU contsraints during training.

---

#### Hyperparameters

1. **Model**

   * `model_id`: `"unsloth/llava-1.5-7b-hf-bnb-4bit"`
   * `lora_r`: `16`
   * `lora_alpha`: `32`
   * `lora_dropout`: `0.05`
   * `bias`: `"none"`
   * `target_modules`: `["q_proj", "v_proj"]`
   * `task_type`: `"CAUSAL_LM"`

2. **Data Processing**

   * `tokenizer.pad_token`: `tokenizer.eos_token` if it was not present
   * `max_length`: `1024` for tokenization

3. **Training Arguments**

   * `output_dir`: `"/content/drive/MyDrive/llava_lora_finetune"`
   * `per_device_train_batch_size`: `1`
   * `per_device_eval_batch_size`: `1`
   * `gradient_accumulation_steps`: `4`
   * `num_train_epochs`: `3`
   * `max_steps`: `100`
   * `learning_rate`: `5e-5`
   * `logging_dir`: `"./logs"`
   * `save_strategy`: `"steps"`
   * `save_steps`: `10`
   * `eval_strategy`: `"no"`
   * `optim`: `"paged_adamw_8bit"`
   * `fp16`: `True`
   * `bf16`: `False`
   * `report_to`: `[]` (no reporting)

4. **Trainer Dataset Selection**

   * `train_dataset`: shuffled with `seed=42`, selected range(100)
   * `eval_dataset`: shuffled with `seed=42`, selected range(20)

---

## 2. Evaluation Methodology and Outcomes


#### Qualitative Results

    Link for viewing demo with 2 prompts : https://youtu.be/yNK_ZEyuWx8

* **Human Review:**

  * Most of the outputs i generated are up to the mark because i have tried it over differnet prompt with different situations and different lengths to find inconsistencies wherever possible.
  * You can see the clear results with 2 diverse prompts in the video.
  * The initial sentiment analysis being done by the model for automatic agent selection is also very good and matches my expectations and the flow is beign maintained.
  * The prompt has really helped the model generalize to the downstream taks instead of the dataset on which it was fine-tuned on.
  * Inference time is a slight more than it should be because first we are doing sentiment analysis using LLM and then using defined rules and then deciding which agent to use and then generating th output. Before this feature of automatic agent selection the inference time was significantly less.
  * We can reduce this inference time by using a lightweight LLM or Graph neural networks for this sentiment analysis task maintaining accuracy and less compute requirement.

* **Failure Zones:**

  * Since we are using LLM for scoring the sentiment for finding the needs of the prompt i guess due to uneven scoring, it is not using multiple agents together instead a single for every prompt which should not be the case according to my code. It is getting biased towards a particular agent.
  * I tried to solve this by setting a threshold score that if you have to use this agent then its score should be greater than a fix number but i due to time constraints i have not still been able to ahcieve that but yeah it can be done later since it is very trivial.
  * Also i have notice inconsistency in response sometimes like for a big prompt (description of problem) it just give one liner ouput and it slightly matches with the context. This is not with every large enough prompts but with some prompts only.

---

## 3. Conclusions and Recommendations

* **Strengths:**

  * The hybrid agent selection system is robust and explainable.
  * Fine-tuned LLaVA responds with contextually relevant, emotionally supportive, and agent-appropriate advice.
  * The multi-modal capability (text + images) is a notable advantage.

* **Limitations:**

  * Performance depends on quality and variety of training data like the dataset i used is still not that useful for this downstream task, the results are good bcause of good prompt engineering.
  * Also, the finetuning took for very short amount of epochs so the dataset which slightly matches with the task is still not completely recognized the model.

---

*This report summarizes the data science process for fine-tuning and evaluating Breakup-Sage, an intelligent, explainable breakup recovery assistant.*
