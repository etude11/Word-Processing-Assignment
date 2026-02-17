# Word Processing Assignment

Analysis of reading time data from Natural Stories corpus examining relationships between word length, frequency, and processing time.

## Structure

```
├── analysis.py              # Main analysis script
├── report.pdf              # Complete analysis report
├── report.md               # Report source (Markdown)
├── dataset/                # Natural Stories data
│   ├── processed_RTs.tsv   # Reading time measurements
│   └── freqs/              # N-gram frequency data
├── plot_length_rt.png      # Length vs RT visualization
├── plot_freq_rt.png        # Frequency vs RT visualization
├── hypothesis1_comparison.png
├── hypothesis2_comparison.png
└── pseudo_vs_real_affixed.png
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
nltk
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn nltk
```

Or on Debian/Ubuntu:
```bash
sudo apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn python3-scipy python3-sklearn python3-nltk
```

## Usage

```bash
python3 analysis.py
```

## Results

### Part I: Correlations
- Length vs Frequency: r=-0.71
- Length vs RT: r=0.31
- Frequency vs RT: r=-0.25

### Part II: Hypothesis Testing
- LM probabilities outperform word frequency (R²=0.102 vs 0.098)
- Content words: contextual predictability better
- Function words: raw frequency better

### Part III: FOBS Model
- Surface frequency marginally better than lemma frequency
- Pseudo-affix hypothesis not confirmed in small sample

See report.pdf for complete analysis.
