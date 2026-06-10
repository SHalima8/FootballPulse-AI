# ⚽ FootballPulse AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit%2DLearn%2C%20XGBoost-orange.svg)](https://scikit-learn.org/)

### 🎯 AI-Powered FIFA World Cup Match Prediction & Tournament Intelligence Platform

**FootballPulse AI** is an end-to-end machine learning project that predicts FIFA World Cup match outcomes and tracks live World Cup news and sentiment in a single interactive dashboard.

Built using **92 years of World Cup history (1930–2022)**, the project combines feature engineering, Elo ratings, predictive modeling, and real-time football intelligence to provide data-driven insights for World Cup matches.

---

## 🚀 Quick Links

- **Live Demo:** Coming soon
- **GitHub Repository:** This repository
- **Notebook Walkthrough:** See `notebooks/` directory for detailed analysis

---

## 🎯 Project Overview

FootballPulse AI answers two key questions:

### 1. Who is most likely to win a match?

Predicts win, draw, and loss probabilities for any World Cup matchup.

### 2. Why does the model think so?

Shows the key factors influencing the prediction, including Elo ratings, attacking strength, defensive strength, and recent form.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/FootballPulse-AI.git
cd FootballPulse-AI
```

2. **Create a virtual environment** (recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n football-pulse python=3.8
conda activate football-pulse
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional, for news API features)
```bash
# Create a .env file in the project root
touch .env
# Add your API keys if using NewsAPI features
# NEWSAPI_KEY=your_api_key_here
```

### Quick Start

**Run the Streamlit application:**
```bash
streamlit run src/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

**Train the model:**
```bash
python src/train_model.py
```

**Make predictions:**
```bash
python src/predict.py
```

---

## 📖 Usage

### Interactive Dashboard

The Streamlit app provides the following features:

1. **Match Prediction** - Select two teams and get win/draw/loss probabilities
2. **News & Sentiment** - View trending World Cup topics and news sentiment

### Python API

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/random_forest_model.pkl')

# Prepare features (19 features as per feature engineering)
match_features = pd.DataFrame({...})

# Make prediction
prediction = model.predict(match_features)
probabilities = model.predict_proba(match_features)
```

---

## 📊 Project Components

### 1. Data Pipeline
- **Raw data**: Located in `data/raw/`
  - `Match_2018_2022.csv` - Recent match data
  - `WorldCupMatches.csv` - Historical matches
  - `WorldCupPlayers.csv` - Player information
  - `WorldCups.csv` - Tournament metadata

- **Processed data**: Located in `data/processed/`
  - `clean_matches.csv` - Cleaned and integrated data
  - `model_features.csv` - Engineered features ready for modeling

### 2. Notebooks
- `01_eda.ipynb` - Exploratory Data Analysis
- `02_featureEngineering.ipynb` - Feature engineering and selection

### 3. Source Code
- `train_model.py` - Model training pipeline
- `predict.py` - Prediction module
- `streamlit_app.py` - Interactive web dashboard

---

##  Key Features

### 🤖 Match Prediction Engine

* Predicts Team A Win / Draw / Team B Win probabilities
* Neutral-ground prediction logic
* Bidirectional probability averaging to eliminate home/away bias
* Interactive match selection dashboard

### 📊 Explainable AI Insights

* Elo rating comparison
* Attack and defense strength analysis
* Goal difference metrics
* Recent form comparison
* Historical performance indicators

### ⚔️ Head-to-Head Analysis

* Historical H2H matchup statistics
* Past performance between selected teams
* Win-loss-draw records in direct matchups
* Tactical insights from previous encounters
* Comparative strength metrics from H2H history

### 📰 Match Intelligence Dashboard

* Trending World Cup topics
* Latest World Cup headlines
* News sentiment classification
* Automatic fallback systems for reliable news retrieval

### 🌍 Live Media Sentiment Dashboard

* Real-time sentiment analysis of breaking news
* Media coverage tracking across multiple sources
* Overall tournament sentiment visualization
* Sentiment trends for teams and matches
* Live headline aggregation with sentiment scoring

---

## 📂 Dataset

### Source

FIFA World Cup historical dataset from Kaggle, extended with manually collected 2018 and 2022 World Cup match data.

### Final Dataset

| Metric              | Value       |
| ------------------- | ----------- |
| Time Span           | 1930 – 2022 |
| Total Matches       | 980         |
| Features            | 19          |
| Target Classes      | 3           |
| Tournaments Covered | 22          |

### Target Classes

* Team A Win
* Draw
* Team B Win

---

## 🧠 Feature Engineering

One of the main goals of this project was building realistic features without data leakage.

### Team Strength Features

* Elo Rating
* Elo Difference
* Elo Ratio
* Attack Strength
* Defensive Strength
* Goal Difference

### Relative Strength Features

* Attack Difference
* Defense Difference
* Goal Difference Difference

### Form Features

* Recent Form (weighted last 5 matches)
* Historical Win Rate
* Form Difference

### Match Context Features

* Tournament Stage Encoding
* Knockout Importance

All features were calculated using only information available before each match occurred.

---

## 📈 Model Development

Three machine learning models were evaluated:

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | 58.67%        |
| Random Forest       | 57.65%        |
| XGBoost             | 59.69%        |

### Final Production Model

**Random Forest Classifier**

Although XGBoost achieved slightly higher accuracy, it exhibited significant overfitting. Random Forest provided the best balance between performance, stability, and generalization.

### Cross Validation

* CV Mean Accuracy: 55.71%
* CV Standard Deviation: 2.81%

This consistency indicated reliable model behavior across different splits.

---

## 📊 Model Performance

### Classification Performance

| Class      | Precision | Recall |
| ---------- | --------- | ------ |
| Draw       | 0.33      | 0.11   |
| Team A Win | 0.61      | 0.91   |
| Team B Win | 0.53      | 0.23   |

Football outcomes are inherently noisy, especially draws between evenly matched teams. Even professional football prediction systems rarely achieve accuracy beyond 60–65%.

---

## 🔍 Technical Challenges Solved

### Challenge 1: Historical Data Gaps

The original dataset ended in 2014.

**Solution**

* Added complete 2018 and 2022 World Cup data manually
* Expanded dataset to 980 matches

---

### Challenge 2: Data Leakage

Several commonly used football statistics accidentally include future information.

**Solution**

* Rebuilt all historical features chronologically
* Ensured each match only used information available at prediction time

---

### Challenge 3: Penalty Shootouts

World Cup knockout matches often end in penalties.

**Solution**

* Extracted winners correctly from penalty shootout records
* Preserved realistic tournament outcomes

---

### Challenge 4: Prediction Bias

Traditional models often treat one team as "home" and another as "away."

**Solution**

* Introduced bidirectional neutral prediction
* Predicted A vs B and B vs A
* Averaged both probabilities

This better reflects actual World Cup conditions where matches occur on neutral grounds.

---

### Challenge 5: Tournament Uncertainty

Single predictions do not capture tournament randomness.

**Solution**

* Built probabilistic tournament simulation
* Added Monte Carlo analysis
* Allowed realistic upset scenarios

---

## ⚠️ Current Limitations

* Historical dataset size remains relatively small compared to modern football datasets.
* Player injuries and squad selections are not incorporated.
* Team form outside World Cup competitions is not included.
* Draw prediction remains the most difficult classification task.

These limitations represent opportunities for future improvement.

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Scikit-Learn
* XGBoost
* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib

### Web Application

* Streamlit

### NLP & News Analysis

* VADER Sentiment Analysis
* NewsAPI
* RSS Feeds

### Development Tools

* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
FootballPulse-AI/
│
├── data/
│   ├── raw/
│   │   ├── Match_2018_2022.csv
│   │   ├── WorldCupMatches.csv
│   │   ├── WorldCupPlayers.csv
│   │   └── WorldCups.csv
│   └── processed/
│       ├── clean_matches.csv
│       └── model_features.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_featureEngineering.ipynb
│
├── src/
│   ├── streamlit_app.py      # Main interactive dashboard
│   ├── train_model.py        # Model training pipeline
│   └── predict.py            # Prediction module
│
├── models/
│   └── (trained model artifacts)
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── README.md
```

---

## 🖼️ Screenshots

### Match Prediction Dashboard
![Dashboard](screenshots/image.png)


### Tournament Simulator
![Simulator](screenshots/image-1.png)

### Monte Carlo Championship Analysis
![Monte-Carlo](screenshots/image-2.png)
![MOnte carlo](screenshots/image-3.png)

## 📚 What I Learned

This was my first fully independent machine learning project.

Key takeaways included:

* End-to-end ML pipeline development
* Feature engineering for sports analytics
* Data leakage prevention
* Model evaluation and overfitting diagnosis
* Explainable AI design
* Interactive dashboard development
* Deployment and project presentation

---

## � Future Improvements

- [ ] Incorporate player-level statistics and squad information
- [ ] Include team form from qualifiers and recent friendlies
- [ ] Add player injury/availability tracking
- [ ] Improve draw prediction accuracy
- [ ] Deploy live dashboard with real-time updates
- [ ] Create REST API for predictions
- [ ] Add betting odds integration
- [ ] Implement ELO ranking historical analysis
- [ ] Add match analytics and tactical insights
- [ ] Reimplement tournament simulation with enhanced features

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
```bash
git clone https://github.com/yourusername/FootballPulse-AI.git
```

2. **Create a feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Commit your changes**
```bash
git commit -m 'Add amazing feature'
```

4. **Push to the branch**
```bash
git push origin feature/amazing-feature
```

5. **Open a Pull Request**

### Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Update README if adding new features
- Test your changes before submitting PR

---

## ❓ FAQ

**Q: How accurate are the predictions?**
A: The model achieves ~60% accuracy on historical World Cup matches. This is reasonable for football predictions, as the sport has inherent randomness. Professional prediction models rarely exceed 65% accuracy.

**Q: Can I use this for betting?**
A: While the model makes reasonable predictions, betting should never be based on a single model. This is an educational project demonstrating ML concepts, not a betting system.

**Q: How is the tournament simulation different from single match prediction?**
A: Single matches use the trained classifier. Tournament simulation uses match probabilities in a bracket format, running 100+ simulations to estimate championship probabilities.

**Q: Why is draw prediction so low?**
A: Draws are difficult because they require teams to be perfectly balanced. With only 980 historical matches and draws being relatively rare, the model struggles with this class.

**Q: Can I use this for other football leagues?**
A: Yes! The framework can be adapted. You'd need to:
1. Collect historical data for your league
2. Re-engineer features using league-specific patterns
3. Retrain the model

---

## 📊 Data Sources

- **Kaggle**: FIFA World Cup historical dataset
- **NewsAPI**: Real-time football news (optional)
- **Manual Collection**: 2018 & 2022 World Cup data

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Sadia Halima**

- Sophomore AI Undergraduate
- Machine Learning & Data Science Enthusiast
- Sports Analytics Passionate

### Interests
- Machine Learning
- Data Science
- Sports Analytics
- AI Product Development

### Connect
- GitHub: [@your-github](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Kaggle for the FIFA World Cup dataset
- Streamlit for the amazing dashboard framework
- Scikit-learn and XGBoost communities
- All contributors and feedback providers

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the FAQ section above
2. Review the notebook tutorials
3. Open an GitHub issue
4. Contact the author

---

**Built with ❤️ for football analytics**
