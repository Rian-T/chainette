from chainette.utils.dag import build_rich_tree, RenderOptions
from chainette.core.branch import Branch
from chainette import Chain, apply


def _fn(x):
    return [x]

node = apply(_fn)  # type: ignore[arg-type]
node.emoji = "📄"  # manually attach icon for test
branch = Branch(name="b", steps=[node])
chain = Chain(name="unicode", steps=[[branch, branch]])


def test_unicode_tree(capsys):
    opts = RenderOptions(icons_on=True, max_branches=2)
    from rich.console import Console
    console = Console(force_terminal=True, width=60)
    console.print(build_rich_tree(chain, opts=opts))

    out = capsys.readouterr().out
    # Expect emoji character present (bullet maybe document icon placeholder)
    assert '📄' in out or '🤖' in out or '🪢' in out 