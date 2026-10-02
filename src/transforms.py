"""MMP transformation rules applied as chemical reactions.

A rule is a reaction SMARTS of the form ``left>>right`` with attachment points ``[*:n]``.
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolops

RDLogger.DisableLog("rdApp.*")

RULE_SEPARATOR = ">>"
HYDROGEN = "[*:1][H]"
ATTACHMENT_POINT = "[*:1]"
MAX_CANDIDATES_PER_REACTION = 10
MAX_PRODUCTS_PER_REACTION = 1


class MMPTransformer:
    def __init__(self, rules: list[str]):
        self.reactions = [reaction for rule in rules if (reaction := self._reaction(rule)) is not None]

    def products(self, mol: Chem.Mol, rng: np.random.RandomState) -> list[Chem.Mol]:
        reactant = Chem.MolToSmiles(mol)
        products = []
        for reaction in self.reactions:
            if not reaction.IsMoleculeReactant(mol):
                continue
            candidates = self._distinct(self._sanitized(self._outcomes(reaction, mol)))
            if len(candidates) > MAX_PRODUCTS_PER_REACTION:
                rng.shuffle(candidates)
                candidates = candidates[:MAX_PRODUCTS_PER_REACTION]
            products.extend(product for product in candidates if Chem.MolToSmiles(product) != reactant)
        return products

    @classmethod
    def _reaction(cls, rule: str) -> AllChem.ChemicalReaction | None:
        left, right = rule.split(RULE_SEPARATOR)
        # Rules that replace a hydrogen atom with a substituent are not applied.
        if left == HYDROGEN:
            return None
        if right == HYDROGEN:
            right = ATTACHMENT_POINT
        smarts = f"{cls._with_explicit_hydrogens(left)}{RULE_SEPARATOR}{cls._with_explicit_hydrogens(right)}"
        try:
            reaction = AllChem.ReactionFromSmarts(smarts)
            reaction.Initialize()
        except ValueError:  # RDKit raises ValueError for rules that cannot be parsed or initialized.
            return None
        return reaction

    @staticmethod
    def _with_explicit_hydrogens(fragment: str) -> str:
        return Chem.MolToSmiles(Chem.MolFromSmiles(fragment), allHsExplicit=True)

    @staticmethod
    def _outcomes(reaction: AllChem.ChemicalReaction, mol: Chem.Mol) -> list[Chem.Mol]:
        outcomes = reaction.RunReactants([mol], maxProducts=MAX_CANDIDATES_PER_REACTION)
        return [product for outcome in outcomes for product in outcome]

    @staticmethod
    def _sanitized(mols: list[Chem.Mol]) -> list[Chem.Mol]:
        return [mol for mol in mols if MMPTransformer._is_sanitizable(mol)]

    @staticmethod
    def _is_sanitizable(mol: Chem.Mol) -> bool:
        try:
            rdmolops.SanitizeMol(mol)
        except ValueError:  # RDKit raises ValueError for products that cannot be sanitized.
            return False
        return True

    @staticmethod
    def _distinct(mols: list[Chem.Mol]) -> list[Chem.Mol]:
        # One molecule for each SMILES, sorted by SMILES so that the random choice in products() is reproducible.
        _, first_indices = np.unique([Chem.MolToSmiles(mol) for mol in mols], return_index=True)
        return [mols[index] for index in first_indices]
