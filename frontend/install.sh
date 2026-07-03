#!/bin/bash

echo "================================================================================"
echo "  Installation du Dashboard Next.js"
echo "================================================================================"
echo ""

echo "[1/3] Installation des dépendances..."
npm install

if [ $? -ne 0 ]; then
    echo ""
    echo "ERREUR: L'installation a échoué"
    exit 1
fi

echo ""
echo "[2/3] Vérification de la configuration..."
if [ ! -f .env.local ]; then
    echo "ERREUR: Fichier .env.local manquant"
    exit 1
fi

echo ""
echo "[3/3] Démarrage du serveur de développement..."
echo ""
echo "================================================================================"
echo "  Dashboard démarré sur http://localhost:3000"
echo "  Appuyez sur Ctrl+C pour arrêter"
echo "================================================================================"
echo ""

npm run dev
