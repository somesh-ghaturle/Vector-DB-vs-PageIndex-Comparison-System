from django import forms


class BenchmarkForm(forms.Form):
    """Form for entering queries and benchmark settings. PDFs handled via raw HTML."""

    queries = forms.CharField(
        label="Search Queries (one per line)",
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 6,
            "placeholder": "data preprocessing steps\nsupervised learning algorithms\nneural network architecture",
        }),
    )
    relevant_pages = forms.CharField(
        label="Relevant Pages (optional — one JSON list per line, matching query order)",
        required=False,
        widget=forms.Textarea(attrs={
            "class": "form-control",
            "rows": 6,
            "placeholder": "[3, 4]\n[7]\n[10, 11]",
        }),
    )
    top_k = forms.IntegerField(
        label="Top-K Results",
        initial=5,
        min_value=1,
        max_value=50,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    num_runs = forms.IntegerField(
        label="Timed Runs per Query",
        initial=20,
        min_value=1,
        max_value=100,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
