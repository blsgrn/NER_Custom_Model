import spacy
import random
from spacy.training.example import Example
from spacy.training import offsets_to_biluo_tags
from spacy.util import minibatch

# Load the pre-trained SciSpaCy model
nlp = spacy.load("en_core_sci_md")

# Define custom labels (e.g., Drug, Dosage, etc.)
CUSTOM_LABELS = ["DRUG", "DOSAGE", "FREQUENCY"]

# Get the Named Entity Recognition (NER) component
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add custom labels to the NER component
for label in CUSTOM_LABELS:
    ner.add_label(label)

# Prepare training data
TRAINING_DATA = [
    ("Prescription for Paracetamol 500mg, take twice daily",
     {"entities": [(19, 28, "DRUG"), (29, 34, "DOSAGE"), (35, 49, "FREQUENCY")]}),
    ("Aspirin 100mg should be taken once a day.",
     {"entities": [(0, 7, "DRUG"), (8, 13, "DOSAGE"), (34, 45, "FREQUENCY")]}),
    ("Ibuprofen 200mg, take every 6 hours.",
     {"entities": [(0, 9, "DRUG"), (10, 15, "DOSAGE"), (22, 33, "FREQUENCY")]}),
]

# Disable other pipes except 'ner' for efficiency
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()

    # Training loop
    for i in range(30):  # Reduce iterations for efficiency
        random.shuffle(TRAINING_DATA)
        losses = {}
        batches = minibatch(TRAINING_DATA, size=2)
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {i + 1}, Losses: {losses}")

# Save the fine-tuned model
nlp.to_disk("/home/balasegaran/PycharmProjects/Custom_NLP_Models/custom_medical_model")

print("Fine-tuned model saved successfully!")
