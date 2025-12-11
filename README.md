# Algorithmische Analyse linearer Klassifikatoren: Perzeptron & ADALINE

## Projektübersicht

Dieses Repository dokumentiert die "First-Principles"-Implementierung und Analyse fundamentaler neuronaler Architekturen. Der Fokus liegt auf der mathematischen Modellierung linearer Klassifikatoren (Perzeptron) und adaptiver linearer Neuronen (ADALINE).

Das Projekt untersucht die Konvergenzeigenschaften von **Online-Learning** (Stochastischer Gradientenabstieg) versus **Batch-Learning** anhand synthetischer, logischer Datensätze (AND, OR, NOT) und erweitert den Untersuchungsraum auf erste Konzepte des Reinforcement Learning (Hexapawn).

-----

## Theoretischer Rahmen

### 1\. Das Rosenblatt-Perzeptron

Das Perzeptron modelliert ein biologisches Neuron als binären Klassifikator. Sei $x \in \{0,1\}^n$ der Eingabevektor und $w \in \mathbb{R}^{n+1}$ der Gewichtsvektor (inklusive Bias $w_0$). Die Aktivierungsfunktion $\phi(z)$ ist definiert als Heaviside-Stufenfunktion:

$$
\phi(z) = \begin{cases} 1 & \text{falls } z \ge 0 \\ 0 & \text{falls } z < 0 \end{cases}, \quad \text{mit } z = w^T x = \sum_{i=0}^{n} w_i x_i
$$

Die Entscheidungsgrenze (Decision Boundary) definiert eine Hyperebene, die den Raum in zwei Halbräume teilt. Die Lernregel basiert auf der Fehlerkorrektur bei Fehlklassifikation:

$$
w := w + \Delta w, \quad \Delta w = \eta (y^{(i)} - \hat{y}^{(i)}) x^{(i)}
$$

Wobei $\eta$ die Lernrate, $y$ das wahre Label und $\hat{y}$ die Vorhersage bezeichnet.

### 2\. ADALINE (Adaptive Linear Neuron)

Im Gegensatz zum Perzeptron minimiert ADALINE eine stetige, differenzierbare Kostenfunktion $J(w)$ (Sum of Squared Errors, SSE) *vor* der Anwendung der Schwellenwertfunktion. Dies ermöglicht ein Lernen über den Gradienten der Fehlerfläche.

$$
J(w) = \frac{1}{2} \sum_{i} (y^{(i)} - \phi(z^{(i)}))^2
$$

Die Gewichtsaktualisierung erfolgt mittels Gradientenabstieg (Gradient Descent):

$$
w := w - \eta \nabla J(w) = w + \eta \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}
$$

-----

## Methodik & Implementierung

Die Algorithmen wurden in Python unter Verwendung von `NumPy` für effiziente Vektoroperationen implementiert.

  * **Logische Gatter:** Modellierung der booleschen Funktionen als linear separierbare Probleme zur Validierung der Gewichtsinitialisierung.
  * **Vektorisierung:** Implementierung des Batch-Learnings durch Matrix-Vektor-Multiplikation (`w.T @ X`) zur Reduzierung der Rechenzeit und Erhöhung der numerischen Stabilität.
  * **Vergleich der Optimierer:**
      * *Stochastischer Gradientenabstieg (SGD):* Update der Gewichte nach jedem Sample $x^{(i)}$.
      * *Batch Gradient Descent:* Akkumulation der Gradienten über die gesamte Epoche.

-----

## Ergebnisse & Visualisierung

### 1\. Lineare Separierbarkeit und Entscheidungsgrenzen

Die folgende Abbildung visualisiert die Entscheidungsgrenze (Decision Boundary) des trainierten Modells im 2D-Feature-Space. Es ist erkennbar, dass der Algorithmus erfolgreich eine Hyperebene gefunden hat, die die Klassen linear trennt.

> <img width="531" height="459" alt="image" src="https://github.com/user-attachments/assets/512f226e-7586-4252-b2f9-46d85b648127" />


### 2\. Konvergenzverhalten: Perzeptron vs. ADALINE

Die Analyse zeigt, dass das klassische Perzeptron bei linear separierbaren Daten in endlicher Zeit konvergiert (Perceptron Convergence Theorem). ADALINE hingegen konvergiert asymptotisch gegen das Minimum der Kostenfunktion, was auch bei nicht perfekt separierbaren Daten zu robusten Ergebnissen führt.

-----

## Ausblick

### Reinforcement Learning (Hexapawn)

Das Notebook enthält eine initiale Implementierung eines Agenten für **Hexapawn**. Dies markiert den Übergang von Supervised Learning zu Reinforcement Learning, wobei der Agent eine Policy $\pi(s)$ lernt, um die erwartete kumulative Belohnung (Reward) zu maximieren.

### Verbindung zu Quantum Machine Learning

Die hier untersuchten linearen Klassifikatoren und Optimierungslandschaften bilden die theoretische Grundlage für **Variational Quantum Circuits (VQC)**. In zukünftigen Arbeiten soll untersucht werden, wie der klassische Gewichtsvektor $w$ durch Rotationswinkel $\theta$ in einem parametrisierten Quantenschaltkreis ersetzt werden kann, um den Hilbert-Raum für Klassifikationsaufgaben zu nutzen.

-----

## Verwendung

```bash
# Repository klonen
git clone [repo-url]

# Abhängigkeiten installieren
pip install numpy matplotlib

# Jupyter Notebook starten
jupyter notebook perzeptron.ipynb
```

-----
