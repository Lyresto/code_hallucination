import random

import plotly.graph_objects as go

from constants import factor2desc, type2desc, affection2desc, excluded_desc, factors_color, affections_color
from tools import iter_data


def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return f'rgba({r}, {g}, {b}, {alpha})'


def main():
    fta_desc = [v for v in sorted(factor2desc.values())] + \
               [v for v in sorted(type2desc.values())] + \
               [v for v in sorted(affection2desc.values())]
    fta_desc = [desc for desc in fta_desc if desc not in excluded_desc]

    factor_affection = {x: [] for x in type2desc.values()}
    random.seed(42)

    for item in iter_data():
        for code_label in item['code_labels']:
            tp = code_label['hallucination-type']
            index = random.randint(0, len(code_label['factors']) - 1)
            factor = code_label['factors'][index]
            index = random.randint(0, len(code_label['affections']) - 1)
            affection = code_label['affections'][index]
            if tp in excluded_desc:
                continue
            # if tp == 'C2' and affection == 'A1':
            #     continue
            factor_affection[tp].append((factor, affection))

    type_colors = {
        'R11': '#2471A3',
        'R12': '#5DADE2',
        'C1': '#1E8449',
        'C2': '#27AE60',
        'C3': '#A9DFBF',
        'C5': '#58D68D',
        'K2': '#A3E4D7',
        'K31': '#73C6B6',
        'K32': '#117A65',
        'K33': '#48C9B0'
    }
    type_colors = {type2desc[k]: v for k, v in type_colors.items()}

    colors = {}
    colors.update(factors_color)
    colors.update(type_colors)
    colors.update(affections_color)

    link_cnt = {}
    for t, lst in factor_affection.items():
        for f, a in lst:
            f_idx = fta_desc.index(f)
            t_idx = fta_desc.index(t)
            a_idx = fta_desc.index(a)
            if (f_idx, t_idx) not in link_cnt:
                link_cnt[(f_idx, t_idx)] = 0
            if (t_idx, a_idx) not in link_cnt:
                link_cnt[(t_idx, a_idx)] = 0
            link_cnt[(f_idx, t_idx)] += 1
            link_cnt[(t_idx, a_idx)] += 1

    source = []
    target = []
    value = []

    for (s, t), v in link_cnt.items():
        source.append(s)
        target.append(t)
        value.append(v)

    nodes = dict(
        pad=25,
        thickness=30,
        line=dict(color="rgba(0,0,0,0)", width=0),
        label=fta_desc,
        color=[colors[desc] for desc in fta_desc]
    )

    links = dict(
        source=source,
        target=target,
        value=value,
        color=[hex_to_rgba(colors[fta_desc[s]], alpha=0.2) for s in source]
    )

    annotations = [
        dict(
            x=0.00,
            y=-0.12,
            text="Causes",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=50)
        ),
        dict(
            x=0.50,
            y=-0.12,
            text="Code Hallucinations",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=50)
        ),
        dict(
            x=1.00,
            y=-0.12,
            text="Impacts",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=50)
        )
    ]

    fig = go.Figure(data=[go.Sankey(
        node=nodes,
        link=links
    )])

    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=32,
            weight="bold",
        ),
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=130
        ),
        width=1600,
        height=1200,
        annotations=annotations
    )
    fig.show()
    # fig.write_image('cause_impact_sankey.pdf', format='pdf')


if __name__ == '__main__':
    main()
