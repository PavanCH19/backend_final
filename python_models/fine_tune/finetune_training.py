


# ============================================================================
# train.py - Training module
# ============================================================================

import os
import json
import random
import re
from spacy.training import offsets_to_biluo_tags
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import spacy
from shared_config import KNOWN_SKILLS, SENTENCE_TEMPLATES, MODEL_DIR, MODEL_NAME


def remove_overlapping_entities(entities):
    """Sort by start position, then prefer longer spans"""
    entities = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    cleaned = []
    last_end = -1
    for start, end, label in entities:
        if start >= last_end:
            cleaned.append((start, end, label))
            last_end = end
    return cleaned


def generate_cv_training_data(num_examples):
    """Generate synthetic CV training data with skill entity annotations"""
    TRAIN_DATA = []
    for _ in range(num_examples):
        num_sentences = random.randint(2, 4)
        cv_text = ""
        entities = []
        cursor = 0

        for _ in range(num_sentences):
            selected_skills = random.sample(KNOWN_SKILLS, random.randint(2, 5))
            template = random.choice(SENTENCE_TEMPLATES)
            sentence = template.format(skills=", ".join(selected_skills))
            cv_text += sentence + " "

            for skill in selected_skills:
                start_idx = sentence.find(skill, 0) + cursor
                if start_idx != -1:
                    end_idx = start_idx + len(skill)
                    entities.append((start_idx, end_idx, "SKILL"))

            cursor += len(sentence) + 1

        entities = remove_overlapping_entities(entities)
        TRAIN_DATA.append((cv_text.strip(), {"entities": entities}))

    return TRAIN_DATA


def clean_training_data(nlp, train_data):
    """Validate and clean training data so entities align with spaCy tokens"""
    cleaned_data = []
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        entities = ann.get("entities", [])

        tags = offsets_to_biluo_tags(doc, entities)

        if "-" in tags:
            valid_entities = []
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is not None:
                    valid_entities.append((span.start_char, span.end_char, label))
            entities = valid_entities

        if entities:
            cleaned_data.append((text, {"entities": entities}))

    return cleaned_data


def debug_train_data(train_data):
    """Debug and validate training data for issues"""
    print("🔎 Running TRAIN_DATA checks...\n")

    for i, (text, ann) in enumerate(train_data):
        entities = ann.get("entities", [])
        spans = []

        for start, end, label in entities:
            span = text[start:end]

            if span != span.strip():
                print(f"[Whitespace Issue] Example {i} -> '{span}' in: {text}")

            if text[start:end] != span:
                print(f"[Index Issue] Example {i} -> ({start}, {end}) gives '{span}' but text slice is '{text[start:end]}'")

            for s, e, l in spans:
                if (start < e and end > s):
                    print(f"[Overlap Issue] Example {i} -> '{span}' overlaps with '{text[s:e]}'")
            spans.append((start, end, label))

    print("\n✅ Finished checking TRAIN_DATA.")


def train_skill_ner(TRAIN_DATA, output_dir, n_epochs, model_name=MODEL_NAME):
    """Fine-tune SpaCy NER model to detect SKILL entities and save it"""
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"❌ Model '{model_name}' not found. Downloading...")
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name)

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    if "SKILL" not in ner.labels:
        ner.add_label("SKILL")

    if TRAIN_DATA:
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.resume_training()
            for epoch in range(n_epochs):
                random.shuffle(TRAIN_DATA)
                losses = {}
                batches = minibatch(TRAIN_DATA, size=compounding(4.0, 16.0, 1.5))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    examples = [Example.from_dict(nlp.make_doc(t), a) for t, a in batch]
                    nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
                print(f"Epoch {epoch+1}/{n_epochs} — Losses: {losses}")

        nlp.to_disk(output_dir)
        print(f"✅ Model trained and saved to {output_dir}")

    return nlp


if __name__ == "__main__":
    print("🔄 Generating training data...")
    TRAIN_DATA = generate_cv_training_data(500)

    with open("elaborated_skill_train_data.json", "w") as f:
        json.dump(TRAIN_DATA, f, indent=4)
    print(f"✅ Generated {len(TRAIN_DATA)} training examples")

    print("\n📋 Sample training data:")
    for i in range(min(3, len(TRAIN_DATA))):
        print(TRAIN_DATA[i])

    print("\n🧹 Cleaning training data...")
    nlp_tmp = spacy.blank("en")
    TRAIN_DATA = clean_training_data(nlp_tmp, TRAIN_DATA)
    print(f"✅ Cleaned dataset size: {len(TRAIN_DATA)} samples")

    debug_train_data(TRAIN_DATA)

    print("\n🤖 Training model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    nlp = train_skill_ner(TRAIN_DATA, MODEL_DIR, n_epochs=20)
    print(f"✅ Model ready for prediction!")

