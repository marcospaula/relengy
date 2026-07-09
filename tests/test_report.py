"""O relatório precisa sair idêntico em Markdown e em Word, da mesma fonte."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from relengy.report import Report  # noqa: E402

docx = pytest.importorskip("docx")


@pytest.fixture
def figura(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    p = tmp_path / "fig.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def test_markdown_tem_titulo_tabela_e_imagem(tmp_path, figura):
    df = pd.DataFrame({"beta": [2.857], "eta": [1846.79], "nota": [None]})
    r = (Report("Bomba centrifuga", subtitle="P-101", base_dir=tmp_path)
         .h1("Achados").para("texto").bullets(["um", "dois"])
         .table(df).image(figura, "legenda").pre("saida bruta"))
    md = r.to_markdown(tmp_path / "R.md").read_text()

    assert md.startswith("# Bomba centrifuga")
    assert "*P-101*" in md
    assert "| beta | eta | nota |" in md
    assert "| 2.857 | 1847 | — |" in md          # None vira travessão
    assert "![legenda](fig.png)" in md            # caminho relativo ao .md
    assert "```text" in md


def test_imagem_relativa_e_resolvida_contra_base_dir(tmp_path, figura):
    """No Word a imagem é embutida: o caminho relativo precisa ter sido resolvido."""
    r = Report("t", base_dir=tmp_path).image("fig.png")
    path, _ = r.blocks[0].payload
    assert path.is_absolute() and path.exists()


def test_docx_e_gerado_com_o_mesmo_conteudo(tmp_path, figura):
    df = pd.DataFrame({"a": [1.5], "b": ["x"]})
    r = (Report("Bomba", base_dir=tmp_path)
         .h1("Um").h2("Dois").para("corpo").callout("destaque")
         .bullets(["i"]).table(df, caption="tab").image(figura, "cap"))
    out = r.to_docx(tmp_path / "R.docx")

    doc = docx.Document(str(out))
    texto = "\n".join(p.text for p in doc.paragraphs)
    assert "Bomba" in texto and "Um" in texto and "corpo" in texto and "destaque" in texto
    assert len(doc.tables) == 1
    assert doc.tables[0].rows[0].cells[0].text == "a"
    assert doc.tables[0].rows[1].cells[0].text == "1.5"
    assert len(doc.inline_shapes) == 1  # a figura foi embutida


def test_pipe_em_celula_nao_quebra_a_tabela_markdown(tmp_path):
    df = pd.DataFrame({"x": ["a|b"]})
    md = Report("t", base_dir=tmp_path).table(df).to_markdown(tmp_path / "R.md").read_text()
    linha = next(ln for ln in md.splitlines() if "a" in ln and "|b" in ln)
    assert linha.count("|") == 3  # bordas + o escapado não conta como separador
    assert r"a\|b" in md
