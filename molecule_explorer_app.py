import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import DataStructs
import base64
from itertools import product

# Set page configuration
st.set_page_config(page_title="Molecule StrawMan Explorer", layout="wide")

# Title and introduction
st.title("Molecular Shape-Preserving Heteroatom StrawMan Replacement")
st.markdown("""
This app allows you to:
1. Input a molecule using SMILES notation
2. Generate novel molecules by replacing heteroatoms while preserving shape
3. Filter by logP similarity and structural features
4. Check commercial availability in the ZINC database (beta)
5. Download results as CSV
""")

# Function to download dataframe as CSV
def get_table_download_link(df, filename="molecule_results.csv", text="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
    
# Function to check if replacement is valid based on chemical rules
def is_valid_replacement(mol, atom_idx, new_atom_symbol):
    """Check if the replacement atom would be valid based on chemical rules"""
    # Special case for sulfur: must have even number of bonds (2, 4, or 6)
    if new_atom_symbol == 'S':
        atom = mol.GetAtomWithIdx(atom_idx)
        num_bonds = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        # Check if bond count is even
        if num_bonds % 2 != 0:
            return False
        # Check if bond count is 2, 4, or 6
        if num_bonds not in [2.0, 4.0, 6.0]:
            return False
    
    # All other replacements are allowed for now
    return True

# Function to generate a 3D conformer
def generate_conformer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Remove existing conformers
    mol.RemoveAllConformers()
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # Optimize the structure
    AllChem.MMFFOptimizeMolecule(mol)
    
    return mol

# Function to calculate principal moments of inertia
def calc_principal_moments(mol):
    if mol is None:
        return None
    
    # Get the conformer
    conf = mol.GetConformer()
    
    # Calculate center of mass
    center_of_mass = np.zeros(3)
    total_mass = 0.0
    
    for atom in mol.GetAtoms():
        mass = atom.GetMass()
        pos = conf.GetAtomPosition(atom.GetIdx())
        center_of_mass[0] += mass * pos.x
        center_of_mass[1] += mass * pos.y
        center_of_mass[2] += mass * pos.z
        total_mass += mass
    
    if total_mass > 0:
        center_of_mass /= total_mass
    
    # Construct inertia tensor
    inertia = np.zeros((3, 3))
    
    for atom in mol.GetAtoms():
        mass = atom.GetMass()
        pos = conf.GetAtomPosition(atom.GetIdx())
        
        # Coordinates relative to center of mass
        x = pos.x - center_of_mass[0]
        y = pos.y - center_of_mass[1]
        z = pos.z - center_of_mass[2]
        
        # Fill inertia tensor
        inertia[0, 0] += mass * (y*y + z*z)
        inertia[1, 1] += mass * (x*x + z*z)
        inertia[2, 2] += mass * (x*x + y*y)
        inertia[0, 1] -= mass * x * y
        inertia[0, 2] -= mass * x * z
        inertia[1, 2] -= mass * y * z
    
    # Make symmetric
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    
    # Get eigenvalues (principal moments)
    principal_moments = np.linalg.eigvalsh(inertia)
    
    # Sort in ascending order
    principal_moments = np.sort(principal_moments)
    
    return principal_moments

# Function to calculate PMI-based shape descriptors
def calc_shape_descriptors(principal_moments):
    # Normalize to get shape-only factors
    if np.all(principal_moments > 0):
        norm_pmi = principal_moments / np.max(principal_moments)
        
        # Compute asphericity
        a_val = norm_pmi[0]
        b_val = norm_pmi[1]
        c_val = norm_pmi[2]
        asphericity = 0.5 * ((a_val - b_val)**2 + (b_val - c_val)**2 + (c_val - a_val)**2)
        
        # Compute eccentricity
        eccentricity = 1.0 - (3.0 * norm_pmi[0]) / sum(norm_pmi)
        
        # Compute normalized ratios
        i1_i3 = norm_pmi[0] / norm_pmi[2]
        i2_i3 = norm_pmi[1] / norm_pmi[2]
    else:
        asphericity = 0.0
        eccentricity = 0.0
        i1_i3 = 0.0
        i2_i3 = 0.0
    
    return asphericity, eccentricity, i1_i3, i2_i3

# Function to analyze molecular shape
def analyze_shape(mol):
    if mol is None:
        return None
    
    # Calculate principal moments of inertia
    principal_moments = calc_principal_moments(mol)
    
    # Calculate shape descriptors
    asphericity, eccentricity, i1_i3, i2_i3 = calc_shape_descriptors(principal_moments)
    
    return {
        'Principal Moments': principal_moments,
        'Asphericity': asphericity,
        'Eccentricity': eccentricity,
        'I1/I3': i1_i3,
        'I2/I3': i2_i3
    }

# Function to generate molecule image for display
def mol_to_img(mol, size=(300, 200)):
    if mol is None:
        return None
    
    # Draw molecule in 2D with explicit Kekulize
    mol = Chem.Mol(mol)  # Create a copy to avoid modifying the original
    
    # Ensure we have a 2D molecule
    try:
        # Remove any 3D conformers and compute 2D coords
        mol.RemoveAllConformers()
        AllChem.Compute2DCoords(mol)
    except:
        pass
    
    # Create the drawer and prepare molecule for drawing
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    
    # Configure the drawing options
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.additionalAtomLabelPadding = 0.3
    
    # Draw the molecule
    try:
        drawer.DrawMolecule(mol)
    except:
        # If drawing fails, try again with a kekulized version
        Chem.Kekulize(mol)
        drawer.DrawMolecule(mol)
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg

# Function to identify aromatic rings and atoms that can be modified
def identify_aromatic_rings_and_ch(mol):
    # Use RDKit's FindSSSR
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    
    aromatic_rings = []
    aromatic_ch_atoms = []
    
    for ring in rings:
        # Check if the ring is aromatic
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            # Get ring atoms
            ring_atoms = [idx for idx in ring]
            aromatic_rings.append(ring_atoms)
            
            # Identify C-H atoms in 6-membered aromatic rings that can be changed to N
            if len(ring) == 6:  # Only in 6-membered rings
                for atom_idx in ring:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                        # This is a C-H that could be replaced with N
                        aromatic_ch_atoms.append(atom_idx)
    
    return aromatic_rings, aromatic_ch_atoms

# Function to check if atom is in an aliphatic chain with at least 2 carbons on each side
def is_valid_aliphatic_position(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    
    # Skip aromatic atoms
    if atom.GetIsAromatic():
        return False
    
    # Skip atoms with fewer than 2 neighbors
    if atom.GetDegree() < 2:
        return False
    
    # Explore chains in both directions
    visited = set([atom_idx])
    neighbors = list(atom.GetNeighbors())
    
    # Check if we can find at least 2 carbons in each chain direction
    results = []
    
    for start_neighbor in neighbors:
        # Skip if already visited
        if start_neighbor.GetIdx() in visited:
            continue
        
        # Start a new chain exploration
        chain_visited = set([atom_idx, start_neighbor.GetIdx()])
        carbon_count = 1 if start_neighbor.GetSymbol() == 'C' else 0
        current_atoms = [start_neighbor]
        
        while carbon_count < 2 and current_atoms:
            next_atoms = []
            
            for current_atom in current_atoms:
                for neighbor in current_atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    
                    if n_idx not in chain_visited:
                        chain_visited.add(n_idx)
                        if neighbor.GetSymbol() == 'C':
                            carbon_count += 1
                        next_atoms.append(neighbor)
            
            current_atoms = next_atoms
        
        results.append(carbon_count >= 2)
    
    # Need at least two chains with 2+ carbons each
    return sum(results) >= 2

# Function to calculate MACCS fingerprint similarity
def calculate_fingerprint_similarity(mol1, mol2):
    # Generate MACCS fingerprints
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return similarity

# Function to generate molecules with heteroatom replacements
def generate_analogs(orig_mol, shape_data, max_analogs=10, min_fp_similarity=0.5, max_fp_similarity=0.95, allow_ch_to_n=True):
    if orig_mol is None:
        return []
    
    # Get the original molecule without hydrogens for modification
    mol = Chem.RemoveHs(orig_mol)
    
    # Get original SMILES for direct comparison
    orig_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    
    # Calculate original fingerprint for later comparison
    orig_fp = MACCSkeys.GenMACCSKeys(mol)
    
    # Identify aromatic rings and C-H atoms that can be converted to N
    aromatic_rings, aromatic_ch_atoms = identify_aromatic_rings_and_ch(mol)
    
    # Map atoms to their rings
    atom_to_ring = {}
    for ring_idx, ring in enumerate(aromatic_rings):
        for atom_idx in ring:
            if atom_idx not in atom_to_ring:
                atom_to_ring[atom_idx] = []
            atom_to_ring[atom_idx].append(ring_idx)
    
    # Identify heteroatom positions
    hetero_positions = []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ['N', 'O', 'S']:
            atom_idx = atom.GetIdx()
            
            # Check constraints
            is_aromatic = atom.GetIsAromatic()
            is_valid_aliphatic = is_valid_aliphatic_position(mol, atom_idx) if not is_aromatic else False
            
            hetero_positions.append({
                'index': atom_idx,
                'symbol': atom.GetSymbol(),
                'is_aromatic': is_aromatic,
                'rings': atom_to_ring.get(atom_idx, []),
                'is_valid_aliphatic': is_valid_aliphatic
            })
    
    # Group heteroatoms by ring
    ring_to_heteroatoms = {}
    for pos in hetero_positions:
        if pos['is_aromatic']:
            for ring_idx in pos['rings']:
                if ring_idx not in ring_to_heteroatoms:
                    ring_to_heteroatoms[ring_idx] = []
                ring_to_heteroatoms[ring_idx].append(pos)
    
    # Define replacement options
    replacements = {'N': ['C', 'N', 'O', 'S'], 
                   'O': ['C', 'N', 'O', 'S'], 
                   'S': ['C', 'N', 'O', 'S']}
    
    # Generate valid exchange combinations
    valid_exchanges = []
    
    # 1. For each aromatic ring, generate candidates with exactly one substitution
    for ring_idx, heteroatoms in ring_to_heteroatoms.items():
        for target_atom in heteroatoms:
            for replacement in replacements[target_atom['symbol']]:
                if replacement != target_atom['symbol']:  # Skip if same as original
                    # Check if the replacement is chemically valid
                    if not is_valid_replacement(mol, target_atom['index'], replacement):
                        continue
                    
                    exchange = {
                        'index': target_atom['index'],
                        'original': target_atom['symbol'],
                        'replacement': replacement
                    }
                    valid_exchanges.append([exchange])
    
    # 2. For each valid aliphatic position, generate candidates
    aliphatic_heteroatoms = [pos for pos in hetero_positions if pos['is_valid_aliphatic']]
    for target_atom in aliphatic_heteroatoms:
        for replacement in replacements[target_atom['symbol']]:
            if replacement != target_atom['symbol']:  # Skip if same as original
                # Check if the replacement is chemically valid
                if not is_valid_replacement(mol, target_atom['index'], replacement):
                    continue
                
                exchange = {
                    'index': target_atom['index'],
                    'original': target_atom['symbol'],
                    'replacement': replacement
                }
                valid_exchanges.append([exchange])
    
    # 3. Add C-H to N replacements in aromatic rings if enabled
    if allow_ch_to_n:
        for ch_idx in aromatic_ch_atoms:
            # N replacement is always valid for C-H in aromatic rings
            exchange = {
                'index': ch_idx,
                'original': 'C',
                'replacement': 'N',
                'is_ch_to_n': True
            }
            valid_exchanges.append([exchange])
    
    # Shuffle and limit the number of combinations to try
    import random
    random.shuffle(valid_exchanges)
    valid_exchanges = valid_exchanges[:max(50, max_analogs * 5)]  # Try more than needed as some might fail
    
    # Original logP for comparison
    orig_logp = Descriptors.MolLogP(mol)
    
    analogs = []
    smiles_set = set([orig_smiles])  # Initialize with original SMILES to avoid regenerating it
    
    # Generate molecules based on the valid exchanges
    for exchanges in valid_exchanges:
        if len(analogs) >= max_analogs:
            break
            
        try:
            # Create a new molecule with the modified atoms
            edit_mol = Chem.EditableMol(mol)
            
            # Apply all exchanges in this combination
            for exchange in exchanges:
                edit_mol.ReplaceAtom(exchange['index'], Chem.Atom(exchange['replacement']))
            
            # Get the modified molecule
            new_mol = edit_mol.GetMol()
            
            # Check if the molecule is valid
            Chem.SanitizeMol(new_mol)
            
            # Generate SMILES for the new molecule (use canonical SMILES for consistent comparison)
            smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True, canonical=True)
            
            # Skip if SMILES is identical to original or we've already generated this SMILES
            if smiles in smiles_set:
                continue
                
            # Calculate fingerprint similarity to original
            fp_similarity = calculate_fingerprint_similarity(mol, new_mol)
            
            # Skip if too similar or too different from original
            if fp_similarity > max_fp_similarity or fp_similarity < min_fp_similarity:
                continue
                
            # Verify the molecule is truly different from the original
            # This is a redundant check but helps catch edge cases
            if smiles == orig_smiles:
                continue
            
            # Add hydrogens and generate 3D conformer
            new_mol_3d = Chem.AddHs(new_mol)
            AllChem.EmbedMolecule(new_mol_3d, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(new_mol_3d)
            
            # Calculate shape similarity to original
            try:
                new_shape = analyze_shape(new_mol_3d)
                
                # Calculate shape similarity score (simplified as Euclidean distance)
                shape_similarity = np.sqrt(
                    (new_shape['I1/I3'] - shape_data['I1/I3'])**2 + 
                    (new_shape['I2/I3'] - shape_data['I2/I3'])**2 +
                    (new_shape['Asphericity'] - shape_data['Asphericity'])**2 +
                    (new_shape['Eccentricity'] - shape_data['Eccentricity'])**2
                )
            except:
                # If shape analysis fails, assign a high (bad) similarity score
                shape_similarity = 999.0
            
            # Calculate logP
            logp = Descriptors.MolLogP(new_mol)
            
            # Calculate logP difference as percentage
            logp_diff_percent = abs(logp - orig_logp) / abs(orig_logp) * 100 if orig_logp != 0 else float('inf')
            
            # Record the atom exchanges made
            exchange_description = ", ".join([
                f"{e['original']}{mol.GetAtomWithIdx(e['index']).GetIdx()}{'-H → N' if e.get('is_ch_to_n') else f' → {e['replacement']}'}" 
                for e in exchanges
            ])
            
            # Add to our results and track the SMILES
            analogs.append({
                'SMILES': smiles,
                'Molecule': new_mol,
                'Molecule_3D': new_mol_3d,
                'LogP': logp,
                'LogP_Diff_Percent': logp_diff_percent,
                'Shape_Similarity': shape_similarity,
                'FP_Similarity': fp_similarity,
                'Exchanges': exchange_description
            })
            smiles_set.add(smiles)
            
        except Exception as e:
            # Skip invalid molecules
            continue
    
    # Double check that none of the analogs have the same SMILES as the original
    analogs = [a for a in analogs if a['SMILES'] != orig_smiles]
    
    # Sort by shape similarity (best first)
    analogs.sort(key=lambda x: x['Shape_Similarity'])
    
    # Limit to the requested number
    return analogs[:max_analogs]

# Function to filter analogs by logP similarity
def filter_by_logp(analogs, threshold=10.0):
    return [analog for analog in analogs if analog['LogP_Diff_Percent'] <= threshold]

# Function to check if molecules are in ZINC database
def check_zinc_availability(smiles_list):
    results = {}
    
    for smiles in smiles_list:
        try:
            # ZINC API endpoint for molecule search
            # Using ZINC20 search instead of ZINC15
            url = f"https://zinc20.docking.org/substances/search/?q={smiles}&structure.smiles-search-type=exact"
            response = requests.get(url, timeout=5)  # Add timeout to avoid hanging
            
            if response.status_code == 200:
                # Check if results are found (simplified check)
                if "No matching" in response.text or "No substances" in response.text:
                    results[smiles] = False
                else:
                    results[smiles] = True
            else:
                # Fallback to ZINC15 if ZINC20 fails
                try:
                    url_fallback = f"http://zinc15.docking.org/substances/search/?q={smiles}&structure.smiles-search-type=exact"
                    response_fallback = requests.get(url_fallback, timeout=5)
                    
                    if response_fallback.status_code == 200:
                        if "No matching" in response_fallback.text:
                            results[smiles] = False
                        else:
                            results[smiles] = True
                    else:
                        results[smiles] = "Error"
                except:
                    results[smiles] = "Error"
                
        except Exception as e:
            results[smiles] = "Error"
    
    return results

# Main app logic
def main():
    # Input SMILES
    smiles_input = st.text_input("Enter SMILES notation:", "C1=CC(=CC2=C1C(N(C2)C3CCC(NC3=O)=O)=O)OC(C)C")
    
    if smiles_input:
        # Generate 3D conformer
        with st.spinner("Generating 3D conformer for shape analysis..."):
            mol_3d = generate_conformer(smiles_input)
            
        if mol_3d is None:
            st.error("Invalid SMILES notation. Please enter a valid molecule.")
            return
        
        # Calculate original molecule properties
        mol_2d = Chem.RemoveHs(mol_3d)
        orig_logp = Descriptors.MolLogP(mol_2d)
        
        # Analyze shape
        shape_data = analyze_shape(mol_3d)
        
        # Compute 2D coordinates for visualization
        AllChem.Compute2DCoords(mol_2d)
        
        # Display original molecule info
        st.header("Original Molecule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2D Structure")
            svg = mol_to_img(mol_2d, size=(400, 300))
            st.components.v1.html(svg, height=300)
        
        with col2:
            st.subheader("Properties")
            st.write(f"SMILES: {Chem.MolToSmiles(mol_2d, isomericSmiles=True, canonical=True)}")
            st.write(f"LogP: {orig_logp:.2f}")
            st.write(f"Molecular Weight: {Descriptors.MolWt(mol_2d):.2f}")
            st.write(f"Heteroatoms: {sum(1 for atom in mol_2d.GetAtoms() if atom.GetSymbol() in ['N', 'O', 'S'])}")
            
            # User-adjustable parameters
            st.subheader("Generation Parameters")
            
            # LogP filter threshold slider
            logp_threshold = st.slider(
                "LogP Similarity Threshold (%)", 
                min_value=5.0, 
                max_value=100.0,  # Extended to 50%
                value=30.0,      # Default set to 20%
                step=5.0,
                help="Maximum allowed LogP difference from original molecule (percentage)"
            )
            
            # Fingerprint similarity range sliders
            st.write("Fingerprint Similarity Range:")
            min_fp_sim, max_fp_sim = st.slider(
                "Similarity Range", 
                min_value=0.0, 
                max_value=1.0,
                value=(0.5, 0.95),
                step=0.05,
                help="MACCS fingerprint Tanimoto similarity range (0=completely different, 1=identical)"
            )
            
            # Maximum number of analogs to generate
            max_analogs_count = st.slider(
                "Maximum Number of Analogs", 
                min_value=1, 
                max_value=40, 
                value=20, 
                step=1,
                help="Maximum number of molecules to generate"
            )
            
            # Option to include C-H to N conversion
            allow_ch_to_n = st.checkbox("Allow C-H → N exchanges in aromatic rings", value=True,
                                      help="Enable replacement of C-H with N in 6-membered aromatic rings")
        
        # Generate analogs with heteroatom replacements
        with st.spinner("Generating molecules with heteroatom replacements..."):
            # Generate analogs using user-defined parameters
            analogs = generate_analogs(mol_3d, shape_data, 
                                     max_analogs=max_analogs_count,
                                     min_fp_similarity=min_fp_sim, 
                                     max_fp_similarity=max_fp_sim,
                                     allow_ch_to_n=allow_ch_to_n)
            
        # Filter by logP similarity with user-defined threshold
        filtered_analogs = filter_by_logp(analogs, threshold=logp_threshold)
        
        st.header("Generated Analogs")
        st.write(f"Generated {len(analogs)} molecules, {len(filtered_analogs)} within {logp_threshold}% logP similarity")
        
        # Display notes about the constraints
        constraint_text = f"""
        **Applied Constraints:**
        1. Maximum of {max_analogs_count} molecules generated
        2. Only one heteroatom exchange per aromatic ring
        3. For aliphatic chains, substitutions only allowed where the heteroatom has at least 2 carbon atoms on each side
        4. Sulfur (S) substitutions only allowed at positions with even number of bonds (2, 4, or 6)
        5. Molecules filtered to have MACCS fingerprint similarity between {min_fp_sim:.2f}-{max_fp_sim:.2f} compared to the original
        6. Only molecules with logP within {logp_threshold}% of the original are displayed
        """
        
        if allow_ch_to_n:
            constraint_text += "\n7. C-H → N exchanges allowed in 6-membered aromatic rings"
            
        st.info(constraint_text)
        
        # Display filtered analogs
        if filtered_analogs:
            # Convert to dataframe for easier display
            df_analogs = pd.DataFrame([{
                'SMILES': a['SMILES'],
                'LogP': a['LogP'],
                'LogP_Diff_%': a['LogP_Diff_Percent'],
                'Shape_Similarity': a['Shape_Similarity'],
                'FP_Similarity': a['FP_Similarity'],
                'Exchanges': a['Exchanges']
            } for a in filtered_analogs])
            
            # Sort by shape similarity
            df_analogs = df_analogs.sort_values('Shape_Similarity')
            
            # Display dataframe
            st.dataframe(df_analogs)
            
            # Download button for results
            st.markdown(get_table_download_link(df_analogs), unsafe_allow_html=True)
            
            # Select molecules to display
            num_molecules = min(5, len(filtered_analogs))
            selected_analogs = filtered_analogs[:num_molecules]
            
            # Option to check commercial availability
            check_availability = st.checkbox("Check ZINC database for commercial availability", value=False)
            
            # Display molecule details
            st.subheader("Top Molecules Details")
            
            if check_availability:
                with st.spinner("Checking ZINC database for commercial availability..."):
                    try:
                        availability = check_zinc_availability([a['SMILES'] for a in selected_analogs])
                    except Exception as e:
                        st.error(f"Error checking ZINC database: {str(e)}")
                        availability = {}
                        st.info("ZINC database connectivity may be limited in this app. In a production environment, proper API calls would be implemented.")
            else:
                availability = {}
            
            # Display molecules regardless of availability check
            for i, analog in enumerate(selected_analogs):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"Molecule {i+1}")
                    st.write(f"SMILES: {analog['SMILES']}")
                    st.write(f"LogP: {analog['LogP']:.2f} (Original: {orig_logp:.2f})")
                    st.write(f"Shape Similarity Score: {analog['Shape_Similarity']:.4f}")
                    st.write(f"Fingerprint Similarity: {analog['FP_Similarity']:.4f}")
                    st.write(f"Exchanges: {analog['Exchanges']}")
                    
                    # Availability info - only show if checked
                    if check_availability:
                        avail = availability.get(analog['SMILES'], "Unknown")
                        if avail is True:
                            st.success("Available in ZINC database")
                        elif avail is False:
                            st.warning("Not found in ZINC database")
                        else:
                            st.info("ZINC availability check pending or failed")
                
                with col2:
                    # Display 2D structure - ensure it's properly 2D
                    mol_copy = Chem.Mol(analog['Molecule'])
                    mol_copy.RemoveAllConformers()  # Remove any 3D info
                    AllChem.Compute2DCoords(mol_copy)  # Calculate 2D coords
                    svg = mol_to_img(mol_copy, size=(400, 300))
                    st.components.v1.html(svg, height=300)
        else:
            st.info("No molecules found within the specified logP similarity threshold.")
    
    # Add footer with instructions
    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. Enter a valid SMILES string of your molecule
    2. Adjust the parameters for molecule generation if needed
    3. Check the "Allow C-H → N exchanges" box to enable replacing aromatic C-H with N
    4. The app will generate molecules by replacing atoms while preserving shape
    5. Only molecules with logP within your specified threshold are retained
    6. You can download the results as a CSV file
    7. You can optionally check if top molecules are available in the ZINC database
    
    ### Notes:
    - Shape similarity is calculated based on 3D conformer analysis (lower scores indicate better shape preservation)
    - The ZINC database check is an approximation; in production, a direct API would be used
    - C-H → N exchanges in aromatic rings maintain aromaticity while changing electronic properties
    - Sulfur atoms require even number of bonds (2, 4, or 6) to maintain proper chemistry
    """)

# Run the app
if __name__ == "__main__":
    main()