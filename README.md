# Dockering

docker build -t legaltext:0.1 .
docker tag legaltext:0.1 bambika/legaltext:0.1
docker push bambika/legaltext:0.1

docker run --rm --gpus all -v C:\Users\andras.janko\Documents\LegalTextDecoder\_data:/app/_data -v C:\Users\andras.janko\Documents\LegalTextDecoder\output:/app/output legaltext:0.1 > training_log.txt 2>&1

docker run --it --rm --gpus all -v C:\Users\andras.janko\Documents\LegalTextDecoder\_data:/app/_data -v C:\Users\andras.janko\Documents\LegalTextDecoder\output:/app/output legaltext:0.1 > training_log.txt 2>&1

docker run -it --rm -v C:\Users\andras.janko\Documents\LegalTextDecoder\_data:/app/_data -v C:\Users\andras.janko\Documents\LegalTextDecoder\output:/app/output legaltext:0.1

# LegalTextDecoder

prompt:

I'm working on a project where my task is to create an NLP model to predict the readability/comprehension difficulty of Hungarian Terms and Conditions (ÁSZF - Általános Szerződési Feltételek) paragraphs for average users, rated on a 1-5 scale.

I have these labels:
1 – Nagyon nehezen érthető: Very difficult/incomprehensible - filled with legal jargon, extremely complex sentence structure, unclear meaning, or excessive cross-references
2 – Nehezen érthető: Difficult - requires significant effort, contains technical terms, complex sentences, partially clear even after multiple readings
3 – Többé/kevésbé megértem: Somewhat understandable - contains difficult parts but main ideas are followable with concentration, may have some manageable cross-references
4 – Érthető: Understandable - generally clear, no unnecessarily complex formulations, most details clear on first reading
5 – Könnyen érthető: Easily understandable - simple, clear language, free of jargon, immediately and unambiguously clear

Here are some rows from my training data (_data/final/train.csv):
student_code,json_filename,text,label_text,label_numeric,labeled_at,lead_time_seconds,text_length
A5VHUA,belvarosi_epito_aszf_labeled.json,"1. Szerződés tárgya
A Vállalkozó az ... teljesítéséhez szükséges.",4-Érthető,4,2025-10-09T18:52:15.489409Z,18.673,1296
FA0B9B,FA0B9B_labeling.json,"Elővételben ... pontjai tartalmazzák.",2-Nehezen érthető,2,2025-10-20T16:02:36.927451Z,15.188,1052
G1QFG2,cimkezes.json,"2.1. Az Orvosi Központ kötelezettséget ... szükséges intézkedést megtesz.",4-Érthető,4,2025-10-29T15:02:46.069275Z,37.418,538

And from my test data (_data/final/test.csv):
student_code,source_file,json_filename,task_id,task_inner_id,annotation_id,text,label_text,label_numeric,completed_by,annotation_created_at,annotation_updated_at,task_created_at,task_updated_at,lead_time_seconds
BCLHKC,3daa4838-otp.txt,cimkezes.json,68,68,68,"2. Ügyfélnek tekintendő ... kéri az OTP Bank Nyrt.-től.",2-Nehezen érthető,2,1,2025-10-12T13:06:42.940971Z,2025-10-12T13:06:42.940991Z,2025-10-12T10:47:27.530748Z,2025-10-12T13:06:43.149576Z,26.951
BCLHKC,3daa4838-otp.txt,cimkezes.json,69,69,69,"4. Az OTP Bank Nyrt. ... ennek megfelelően értelmezendő.",2-Nehezen érthető,2,1,2025-10-12T13:07:10.082983Z,2025-10-12T13:07:10.083007Z,2025-10-12T10:47:27.530830Z,2025-10-12T13:07:10.292287Z,25.074

The result of my baseline model:

Baseline Model: DummyClassifier (Most Frequent Strategy)
Approach: The baseline model always predicts class 4 (Érthető - Understandable), which is the most common label in the training data, appearing in 31.83% of training samples.

Performance on Test Set:
- Accuracy: 20.45% (27 out of 132 predictions correct)
- F1-Macro Score: 0.0679
- F1-Weighted Score: 0.0695

The model exclusively predicts class 4, completely ignoring all other readability levels (classes 1, 2, 3, and 5). This results in 0% precision and recall for all non-majority classes. The 20.45% accuracy reflects the proportion of class 4 samples in the test set. This is actually lower than the training distribution (31.83%), indicating the test set is more balanced across all five classes. The confusion matrix shows 27 correct predictions (all class 4 samples) and 105 incorrect predictions, with all errors being false positives for class 4.
