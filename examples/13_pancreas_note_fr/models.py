"""Data models for the Pancreas Cancer French Clinical Note Generator pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClinicalCase(BaseModel):
    """Input model for clinical case text."""
    text: str = Field(..., description="Text of the clinical case")


class ClassifiedCase(BaseModel):
    """Clinical case with classification category."""
    category: str = Field(..., description="pancreas | lymphoma | other")


class ClinicalNote(BaseModel):
    """Reformulated clinical note."""
    note: str = Field(..., description="Reformulated clinical note")


class ExtractedInfo(BaseModel):
    """Structured information extracted from clinical notes."""
    
    # Patient demographics
    sexe: Literal["Homme", "Femme"] | None = Field(None, description="Sexe du patient")
    age_diagnostic: int | None = Field(None, description="Âge au diagnostic (années)")
    date_naissance: str | None = Field(None, description="Date de naissance du patient")
    date_diagnostic: str | None = Field(None, description="Date du diagnostic")
    ethnie: str | None = Field(None, description="Ethnicité si pertinente")

    # Disease characteristics
    type_histologique: str | None = Field(None, description="Type histologique")
    sous_type_ihc: str | None = Field(None, description="Sous-type immunohistochimique")
    stade_ann_arbor: str | None = Field(None, description="Stade Ann Arbor I-IV")
    localisation_initiale: str | None = Field(None, description="Localisation initiale de la maladie")
    atteinte_medullaire: bool | None = Field(None, description="Atteinte médullaire")
    atteinte_viscerale: bool | None = Field(None, description="Atteinte viscérale")
    symptomes_b: bool | None = Field(None, description="Symptômes B présents")
    bulky: bool | None = Field(None, description="Masse tumorale volumineuse")
    index_performance: int | None = Field(None, description="Index de performance (ECOG 0-5)")

    # Laboratory results
    hemogramme: str | None = Field(None, description="Hémogramme")
    ldh: float | None = Field(None, description="LDH sérique (U/L)")
    beta2_microglobuline: str | None = Field(None, description="β2-microglobuline")
    serologies: str | None = Field(None, description="Sérologies virales")
    bilan_hepato_renal: str | None = Field(None, description="Bilan hépatique et rénal")

    # Treatment
    chimiotherapie: str | None = Field(None, description="Schéma de chimiothérapie")
    cycles: int | None = Field(None, description="Nombre de cycles")
    immunotherapie: str | None = Field(None, description="Immunothérapie utilisée")
    radiotherapie: str | None = Field(None, description="Radiothérapie (site/dose)")
    dates_traitement: str | None = Field(None, description="Dates début/fin traitement")
    reponse: str | None = Field(None, description="Réponse au traitement")

    # Outcomes
    date_rc: str | None = Field(None, description="Date de rémission complète")
    date_rechute: str | None = Field(None, description="Date/site de rechute")
    pfs: float | None = Field(None, description="Survie sans progression (mois)")
    os: float | None = Field(None, description="Survie globale (mois)")
    statut_final: str | None = Field(None, description="Statut à la dernière visite")

    # Additional information
    comorbidites: str | None = Field(None, description="Comorbidités significatives")
    antecedents_fam: str | None = Field(None, description="Antécédents familiaux")
    facteurs_risque: str | None = Field(None, description="Facteurs de risque")
    imagerie: str | None = Field(None, description="Résultats d'imagerie")


class SpanInfo(BaseModel):
    """Text spans for extracted information."""
    
    sexe_span: str | None = None
    age_diagnostic_span: str | None = None
    date_naissance_span: str | None = None
    date_diagnostic_span: str | None = None
    ethnie_span: str | None = None
    type_histologique_span: str | None = None
    sous_type_ihc_span: str | None = None
    stade_ann_arbor_span: str | None = None
    localisation_initiale_span: str | None = None
    atteinte_medullaire_span: str | None = None
    atteinte_viscerale_span: str | None = None
    symptomes_b_span: str | None = None
    bulky_span: str | None = None
    index_performance_span: str | None = None
    hemogramme_span: str | None = None
    ldh_span: str | None = None
    beta2_microglobuline_span: str | None = None
    serologies_span: str | None = None
    bilan_hepato_renal_span: str | None = None
    chimiotherapie_span: str | None = None
    cycles_span: str | None = None
    immunotherapie_span: str | None = None
    radiotherapie_span: str | None = None
    dates_traitement_span: str | None = None
    reponse_span: str | None = None
    date_rc_span: str | None = None
    date_rechute_span: str | None = None
    pfs_span: str | None = None
    os_span: str | None = None
    statut_final_span: str | None = None
    comorbidites_span: str | None = None
    antecedents_fam_span: str | None = None
    facteurs_risque_span: str | None = None
    imagerie_span: str | None = None


# Rebuild models to ensure proper initialization
ExtractedInfo.model_rebuild() 