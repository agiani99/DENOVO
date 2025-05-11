# Molecule Explorer: Shape-Preserving Molecular Substitution Tool

![Molecule Explorer Screenshot](https://github.com/agiani99/DENOVO/blob/main/Screenshot_DenovoAZ.png)

## Overview

Molecule Explorer is an interactive web application that generates novel molecules by systematically replacing heteroatoms while preserving the 3D shape of the original molecule. This tool is designed for medicinal chemists and drug discovery researchers who want to explore chemical space efficiently while maintaining key molecular properties.

## Features

- **Interactive SMILES Input**: Enter any valid SMILES string to analyze and modify
- **Shape-Preserving Substitutions**: Generate molecules that maintain similar 3D shape to the original
- **Multiple Substitution Types**:
  - Heteroatom replacements (N, O, S → C, N, O, S)
  - C-H → N exchanges in 6-membered aromatic rings
- **Chemical Validity Constraints**:
  - Only one substitution per aromatic ring
  - Aliphatic substitutions require 2+ carbon atoms on each side
  - Sulfur substitutions only allowed with even number of bonds (2, 4, or 6)
- **Property Filters**:
  - LogP similarity threshold (adjustable from 5-50%)
  - MACCS fingerprint Tanimoto similarity range
- **Results Visualization**:
  - 2D structure visualization
  - Shape similarity scores
  - LogP comparison
  - Detailed exchange information
- **Export Capabilities**:
  - Download results as CSV
- **Commercial Availability Check**:
  - Optional ZINC database lookup (beta)

## Installation

### Prerequisites

- Python 3.7+
- RDKit
- Streamlit
- NumPy
- Pandas

### Setup

1. Clone the repository:
```bash
git clone https://github.com/agiani99/DENOVO.git
cd DENOVO
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run molecule_explorer_app.py
```

## Usage

1. **Input a Molecule**: Enter a SMILES string in the input field
2. **Adjust Parameters**:
   - LogP Similarity Threshold (%)
   - Fingerprint Similarity Range
   - Maximum Number of Analogs
   - Enable/disable C-H → N exchanges
3. **Generate Analogs**: The app automatically processes and generates molecules
4. **Review Results**: 
   - Table of generated molecules with properties
   - 2D structures of top molecules
   - Download results as CSV
5. **Optional**: Check commercial availability in ZINC database

## Scientific Background

### Shape Similarity

The 3D shape of molecules is analyzed using principal moments of inertia, which provide a rotationally-invariant description of molecular shape. The shape similarity is quantified using:
- Normalized principal moments ratios (I1/I3, I2/I3)
- Asphericity
- Eccentricity

### Bond Constraints

- **Aromatic Systems**: Limited to one substitution per ring to maintain aromaticity
- **Aliphatic Chains**: Require carbon buffers to maintain chain integrity
- **Sulfur Chemistry**: Enforces correct valence states (2, 4, or 6 bonds)

## Applications

- **Scaffold Hopping**: Identify new chemical scaffolds with similar 3D shape
- **Patent Busting**: Generate structurally distinct molecules with similar properties
- **Lead Optimization**: Explore chemical modifications that preserve key interactions
- **Property Tuning**: Fine-tune properties while maintaining overall molecular shape



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Future Developments

- Support for multiple conformer generation
- Additional molecular property filters
- Integration with more chemical databases
- Pharmacophore-based constraints
- Machine learning models to predict activity

## Contact

For questions or suggestions, please open an issue on GitHub or contact [agiani99@hotmail.com].
