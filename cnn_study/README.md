# Cross-hospital TB screening with a compact CNN

The model learns filters directly from chest X-rays. The experiment asks whether a screening cutoff picked in Shenzhen still works in Montgomery County. The submission includes working CPU inference, two trained checkpoints, the case manifest, the Overleaf source and the recorded experiment. The full model is a small two-convolution NumPy network developed for this study. The four-block PyTorch model in the team repository is a separate implementation.

## Results to show in class

The full CNN caught 57 of 58 Montgomery positives but also flagged 77 of 80 normal images. Its external sensitivity was 98.28%, specificity 3.75%, and ROC AUC 0.603. Both models used exactly the same hospital split as the HOG baselines. The dashboard shows all four models at the locked cutoffs and at 0.5. A copy of the public sample radiograph and its reference score are in `sample/`.

| Requirement | Evidence |
|---|---|
| Working implementation on GitHub | Add this whole directory as `cnn_study/` to the existing team repository. Keep the scripts, templates, data manifest, results and trained weights together. |
| Research paper in Overleaf | Import `Overleaf_TB_CNN.zip` as a new Overleaf project. It contains `main.tex` and the confusion-matrix figure. |
| Ablation | The full model uses 35% dropout. The ablation uses 0%. Both runs share initialization, training cases, minibatch order, epoch budget and threshold selection rule. |
| Statistical analysis | Wilson 95% intervals, paired class-stratified bootstrap differences and exact McNemar test. See `results/cnn_metrics.csv`, `results/paired_differences.csv`, and `results/summary.json`. |
| Cross-dataset validation | 496 Shenzhen training, 166 Shenzhen validation, 138 Montgomery external test. Case IDs, hashes and assignments are in `data/manifest.csv`. |

## Deployment, one step at a time

1. Extract the main ZIP. The folder to upload to GitHub is `cnn_study`. The separate Overleaf ZIP is ready to import.
2. Open the team GitHub repository, choose **Add file** then **Upload files**, and drag the extracted `cnn_study` folder from File Explorer onto the upload page. Use the folder itself so GitHub keeps its subfolders. If your browser accepts only individual files, use GitHub Desktop: clone the team repository, copy the folder into the clone, commit, and push. Check that `cnn_study/app.py` and `cnn_study/results/full.npz` both appear online.
3. In Render choose New → Web Service and select the repository. Pick the branch containing the upload.
4. Select Python and set **Root Directory** to `cnn_study`.
5. Set **Build Command** to `pip install -r requirements.txt`.
6. Set **Start Command** to `gunicorn --workers 1 --threads 2 --bind 0.0.0.0:$PORT app:app`. The health-check path is `/health`.
7. Deploy. Choose a service plan appropriate for your account; the demo runs on CPU and reads the shipped checkpoint. Copy the assigned URL once the service is live.
8. Open the URL, switch the cohort dropdown to Shenzhen and back, then upload `sample/public_sample.png`. Compare the result with `sample/reference.json`.
9. In Overleaf choose New Project → Upload Project and import `Overleaf_TB_CNN.zip`. Compile `main.tex`, replace the generic author line with the team's actual names, and read through the methods and results before sharing.
10. Give the evaluator your GitHub repo link, the Overleaf file or view link, and the Render link. Add the actual public links to the team README after they exist.

Render's Flask guide: https://render.com/docs/deploy-flask

## Run the demo locally

From inside this directory, use Python 3.12. On Windows, install the three application libraries with `python -m pip install numpy Pillow Flask`, then run `python -m flask --app app run`. Open the address printed in the terminal. Gunicorn is for the Linux deployment. Images are processed in memory. Use public research X-rays for the demonstration. The score is the model's output for the study label; clinical diagnostic performance and calibration have not been established.

## Reproduce the experiment

Use the Colab notebook in this folder or the commands below. Retraining is optional for deployment. The repository includes fitted weights.

```sh
python prepare_data.py "/path/to/Kaggle-archive.zip" --out image_cache.npz
python experiment.py image_cache.npz --out rerun
```

The archive or extracted directory must contain `Lung Segmentation/CXR_png/`. Set `OPENBLAS_NUM_THREADS=1` before running Python for small-matrix CPU performance. The original run used Python 3.12, NumPy 2.3.5, a fixed seed of 42 and 20 epochs per variant. Different BLAS implementations can produce small numerical differences. The included demo and paper refer to the frozen supplied run. The Colab notebook saves repeat-run outputs separately.

The original results CSVs for HOG and HOG + PCA come from the team's previous run on the same case manifest. The earlier Montgomery outcomes were already known. Treat this as an exploratory cross-hospital study. The notebook and scripts expect the public Kaggle data at https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels.
