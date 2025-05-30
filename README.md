# Retinal-Vessel-Segmentation-Using-Deep-Learning-Techniques
Biomedical Imaging and Analytics

This project explores and compares deep learning models for segmenting retinal blood vessels in fundus images using the DRIVE dataset. The goal is to improve diagnostic assistance in detecting retinal diseases such as diabetic retinopathy and hypertension.


---

## Evaluation Metrics

We used the following metrics to evaluate model performance:

- **Accuracy**
- **F1 Score**
- **Jaccard Index (IoU)**
- **Precision**
- **Recall**
- **Dice Coefficient**


---

## 🧠 Model Descriptions

- **U-Net**: A classic encoder-decoder architecture with skip connections, designed for biomedical image segmentation.
- **nnU-Net**: A robust, self-configuring version of U-Net that automatically adapts to any new biomedical dataset.
- **Swin UNet**: A hybrid model combining U-Net with Swin Transformer blocks, allowing hierarchical attention and better global context.
- **SAM (Segment Anything Model)**: A large-scale foundation model originally trained for general-purpose segmentation, adapted here for medical imaging.

---





---

## Results Summary

| Model        | Accuracy | F1 Score | Jaccard | Precision | Recall | Dice Coef |
|--------------|----------|----------|---------|-----------|--------|-----------|
| **U-Net**     | 0.96429  | 0.78742  | 0.64983 | 0.81934   | 0.76219| 0.75560   |
| **nnU-Net**   | 0.96877  | 0.81656  | 0.68756 | 0.84589   | 0.78921| 0.81468   |
| **SAM**       | 0.96162  | 0.71408  | 0.55668 | 0.80539   | 0.69379| 0.74830   |
| **Swin UNet** | 0.95270  | 0.80260  | 0.62123 | 0.84360   | 0.78143| 0.76623   |

We evaluated four segmentation models — U-Net, nnU-Net, Swin UNet, and a fine-tuned Segment Anything Model (SAM) — on the DRIVE retinal vessel dataset. 

- **nnU-Net** achieved the best overall performance with an F1 score of **0.8165** and a Dice coefficient of **0.8147**, showing its strong adaptability and automated configuration benefits.
- **Swin UNet** performed well with competitive precision (**0.8436**) and recall (**0.7814**), leveraging transformer-based attention mechanisms.
- **U-Net** delivered stable results and maintained good balance across metrics, making it a reliable baseline.
- **SAM**, while promising in generalization, underperformed on this small medical dataset (F1: **0.7141**) due to its lack of domain-specific fine-tuning and the limited training data.

Overall, the results highlight the trade-off between generalization (SAM) and task-specific performance (nnU-Net), especially in the context of biomedical image segmentation.


---

## Limitations and Future Work

- **Small Dataset**: The DRIVE dataset contains only 40 images, limiting model generalization.
- **SAM Generalization Gap**: SAM performs well in generic tasks but requires adaptation for fine vessel structures.
- **Dataset Diversity**: Future work should evaluate models on datasets like CHASE_DB1 or STARE.
- **Hybrid Potential**: Combining transformer-based encoders with classic UNet-style decoders can yield strong results.

---

## Novelty and Impact

This project integrates and benchmarks a wide range of segmentation architectures, including cutting-edge models like SAM and Swin-UNet. Our results highlight the importance of adapting general-purpose vision models to domain-specific tasks such as retinal vessel segmentation.

---

## 📎 Citation

If you use this codebase or findings, please cite the respective works for U-Net, nnU-Net, Swin UNet, and SAM.

---
