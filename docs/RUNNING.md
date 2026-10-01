# Run Orsay · Ejecutar Orsay

## English

GitHub stores the code and version history. Viewing a repository does not run Python. Choose one of the following options.

### Your laptop (recommended for regular development)

Install Python 3.11 or later. In GitHub choose **Code → Download ZIP**, extract it and open a terminal in the extracted folder. On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples/offline_demo.py
```

On macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/offline_demo.py
```

The explicit environment Python path avoids shell activation-policy issues. The demo runs on fictional teams and produces `outputs/demo/synthetic_predictions.csv` and `synthetic_metrics.csv`. No API token is needed.

For ongoing changes, prefer `git clone` over repeated ZIP downloads. After publication, use the exact repository URL shown in GitHub's **Code** menu. A clone preserves Git history and supports `git pull`, branches and commits; a ZIP is a file snapshot without that history.

### In your browser

Open the repository and choose **Code → Codespaces → Create codespace on main**. In its terminal run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/offline_demo.py
```

The Codespace opens at the root of this standalone repository; run the commands above there.

Codespaces executes Python in a remote development environment. It has included usage quotas and may incur charges beyond them if billing is enabled. Check your account usage; stop the Codespace when finished. This project does not create one automatically.

### Historical data

The recovered full workflow lives in `notebooks/01_mec1_modelo_prepartido_v121.ipynb`. It additionally requires `requirements-full.txt`, locally configured API access and appropriate dataset permissions. It has not been rerun during portfolio preparation. Do not upload tokens or downloaded API datasets. The synthetic demo does not reproduce the full calibration workflow.

## Español

**Repositorio:** guarda código y versiones. **Python:** ejecuta el modelo. **Codespaces:** permite ejecutar Python desde el navegador. **GitHub Pages:** puede publicar HTML/JavaScript estáticos; no ejecuta este modelo Python en el servidor.

Para comenzar, descarga y descomprime el ZIP, abre una terminal dentro de la carpeta y usa los comandos de Windows o macOS/Linux de arriba. Se crea un entorno separado, se instalan las dependencias, se verifican los controles y se ejecuta la demo. Los CSV aparecen en `outputs/demo/`; puedes abrirlos con Excel.

Si vas a desarrollar Orsay continuamente, conviene clonar el repositorio y trabajar desde VS Code. Editar un archivo local no actualiza GitHub: hay que guardar los cambios en un commit y enviarlos con `git push`. Los resultados de la demo son sintéticos y no demuestran precisión predictiva en partidos reales.

## Official references

- [GitHub Codespaces](https://docs.github.com/en/codespaces/about-codespaces/what-are-codespaces)
- [Codespaces billing](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces)
- [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
