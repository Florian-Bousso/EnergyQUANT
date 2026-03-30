"""
Modèle de forecasting des prix énergétiques basé sur Prophet (Meta).

Entraînement, configuration et inférence du modèle Prophet pour la
prévision des séries temporelles de prix de l'énergie. Gère les
saisonnalités propres aux marchés énergétiques (hebdomadaire, annuelle,
jours fériés), les régresseurs externes (température, vent, solaire)
et l'export des prévisions avec intervalles de confiance.
"""
