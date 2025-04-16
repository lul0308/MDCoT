## 📂 Dataset Usage Instructions

Due to privacy restrictions, we cannot directly provide the ADNI dataset used in this project. However, you can reproduce the dataset by following the steps below.

---

### 1️⃣ Request Access

- Visit the [ADNI official website](https://adni.loni.usc.edu/data-samples/adni-data/)
- Apply for access through **IDA (Image & Data Archive)**
- After approval, you will receive a confirmation email

---

### 2️⃣ Download & Filter Data

Once access is granted, log into the ADNI portal and use the **Advanced Search** feature to download multimodal data:

- **🧠 MRI image data**: Download directly using Advanced Search  
- **📄 Clinical text data**: Extract from associated **Metadata** files  
  For details, refer to this guide:  
  [Clinical data extraction guide](https://blog.csdn.net/C_abbage/article/details/118297928)

---

### 3️⃣ Data Preprocessing

#### 🧠 MRI Preprocessing

MRI images are typically in DICOM format and require the following steps:

- Format conversion (DICOM → NIfTI)
- Slice-timing correction
- Motion correction
- Normalization
- Registration
- Smoothing

Recommended reference:  
[MRI preprocessing guide](https://blog.csdn.net/m0_37852937/article/details/117251299)

---

#### 📄 Clinical Text Preprocessing

Extract key fields from metadata (e.g., age, gender, MMSE, FAQTOTAL, CDGLOBAL, NPISCORE), convert them into natural language format, and anonymize any personally identifiable information.

> **Example (natural language format):**  
> `"This is a 71-year-old female. MMSE score is 17.0, CDGLOBAL score is 0.5, NPISCORE is 2.0, FAQTOTAL score is 13.0."`
