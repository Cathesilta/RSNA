# RSNA
Repo for kaggle RSNA Intracranial Hemorrhage Detection.
Final score at 0.06041 stage-2 private leaderboard with evaluation metric Log Loss.
Ref: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

# Background
Intracranial hemorrhage, bleeding that occurs inside the cranium, is a serious health problem requiring rapid and often intensive medical treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient.

Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

In this competition, your challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes.

You’ll develop your solution using a rich image dataset provided by the Radiological Society of North America (RSNA®) in collaboration with members of the American Society of Neuroradiology and MD.ai.

If successful, you’ll help the medical community identify the presence, location and type of hemorrhage in order to quickly and effectively treat affected patients.

Challenge participants may be invited to present their AI models and methodologies during an award ceremony at the RSNA Annual Meeting which will be held in Chicago, Illinois, USA, from December 1-6, 2019.

Collaborators
Four research institutions provided large volumes of de-identified CT studies that were assembled to create the challenge dataset: Stanford University, Thomas Jefferson University, Unity Health Toronto and Universidade Federal de São Paulo (UNIFESP), The American Society of Neuroradiology (ASNR) organized a cadre of more than 60 volunteers to label over 25,000 exams for the challenge dataset. ASNR is the world’s leading organization for the future of neuroradiology representing more than 5,300 radiologists, researchers, interventionalists, and imaging scientists. MD.ai provided tooling and support for the data annotation process.

The RSNA is an international society of radiologists, medical physicists and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for AI to assist in detection and classification of hemorrhages in order to prioritize and expedite their clinical work.

(https://github.com/pengbo0054/RSNA/blob/master/samples/ID_0c5667bea.png)

# Data Structure
```
RSNA
    stage_1
        stage_1_train.csv
        stage_1_sample_submission.csv
        stage_1_train
            ID_000012eat.dcm
            ID_6431af929.dcm(corrupted)
            ...
        stage_1_test
            ID_000000e27.dcm
            ...
    stage_2
        stage_2_train.csv
        stage_2_sample_submission.csv
        stage_2_train
            ID_000039ta0.dcm
            ...
        stage_2_test
            ID_000009146.dcm
            ...
```
# DataFrame Transformation
```
This is refering to the stage_x_train.csv file.
Origin format
	ID	                            Label
0	ID_12cadc6af_epidural	        0
1	ID_12cadc6af_intraparenchymal	0
2	ID_12cadc6af_intraventricular	0
3	ID_12cadc6af_subarachnoid	    0
4	ID_12cadc6af_subdural	        0
5	ID_12cadc6af_any	            0

Transformed format

	name	        epidural	intraparenchymal	intraventricular	subarachnoid	subdural	any
0	ID_12cadc6af	0	        0	                0	                0	            0	        0
1	ID_38fd7baa0	0	        0	                0	                0	            0	        0
2	ID_6c5d82413	0	        0	                0	                0	            0	        0
```
# Data Transform from dicom format to png
data_trainsform.ipynb is to convert dicom file to png
This may take hours to do so.

# Train the network
Just train!
