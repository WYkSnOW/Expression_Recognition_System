import streamlit as st
import pandas as pd

#
st.title('Facial Expression Recognition System Using Machine Learning')

#
st.text('By: Waiyuk Kwong, Zhihui Chen, Tyler Lin, Blane R. York, and Carter D. Robinson')

#
st.header('Introduction/Background')
st.write('Facial expressions play a crucial role in non-verbal communication, conveying emotions and intentions. '
        'This project aims to develop a system that automatically detects and recognizes facial expressions using machine learning techniques.'
        )
st.markdown("""
- **Existing methods** often rely on CNNs (Convolutional Neural Networks) to process images or use facial key points to capture changes in facial geometry.
- **Widely used** in research and everyday applications.
""")

#
st.header('Problem Definition')
st.markdown("""
**Problem:**

- How can facial expressions be accurately recognized in real-time scenarios using facial detection systems?

**Motivation:**

- While current facial expression recognition systems are effective, they often struggle with real-time performance and robustness. By combining facial key points and image data, we aim to develop a more efficient and robust system.
""")

#
st.header('Methods')

#
st.header('Results and Discussion')

#
st.header('References')
st.markdown("""
1. A. Huang and M. J. C. Samonte, "Using Emotion Analysis, Eye tracking, and Head Movement to Monitor Student Engagement among ESL Students with Facial Recognition Algorithm (Mediapipe)," 2024 7th International Conference on Advanced Algorithms and Control Engineering (ICAACE), Shanghai, China, 2024, pp. 509-513, doi: [10.1109/ICAACE61206.2024.10548871](https://doi.org/10.1109/ICAACE61206.2024.10548871).
2. F. M. Sadikoğlu and M. Idle Mohamed, "Facial Expression Recognition Using CNN," 2022 International Conference on Artificial Intelligence in Everything (AIE), Lefkosa, Cyprus, 2022, pp. 95-99, doi: [10.1109/AIE57029.2022.00025](https://doi.org/10.1109/AIE57029.2022.00025).
3. K. Wang, R. Li, and L. Zhao, "Real-time facial expressions recognition system for service robot based-on ASM and SVMs," 2010 8th World Congress on Intelligent Control and Automation, Jinan, 2010, pp. 6637-6641, doi: [10.1109/WCICA.2010.5554164](https://doi.org/10.1109/WCICA.2010.5554164).
""")
