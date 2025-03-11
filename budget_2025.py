# %% Setup
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import holoviews as hv
import plotly.graph_objects as go
import plotly.express as px
from holoviews.core import Store
hv.extension('bokeh')

DIR = '/Users/leodai/Library/CloudStorage/OneDrive-Personal/Work/Blog/5 HK Budget 2025'

# %% Data

# Revenue Summary
revenue_sum_ = pd.read_csv(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/general-revenue-account/sum_ra.csv', encoding= 'ISO-8859-1')
revenue_sum_chi = pd.read_csv(f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/¨F©≤§@ØÎ¶¨§J±b•ÿ/sum_ra.csv', encoding='utf-8')

revenue_internal_wcat = pd.read_csv(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/revenue-analysis-by-head/head003.csv', encoding= 'ISO-8859-1')
revenue_internal_chi = pd.read_csv(f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/¡`•ÿ¶¨§J§¿™R/head003.csv', encoding='utf-8')

# Other Revenue
revenue_other = pd.DataFrame(
    columns=[
        'RevenueCategory', '收入項目', 'rev_25to26_millions'],
    data=[
        ['Government Bond' , '政府債券', 150000],
        ['Deficit', '赤字', 67006],
    ]
)

# Revenue by head
pattern = re.compile(r"head(\d+)\.csv$")
revenue_by_head_files = glob.glob(os.path.join(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/revenue-analysis-by-head', "head*.csv"))
revenue_by_head_dfs = []
for file_path in revenue_by_head_files:
    file_name = os.path.basename(file_path)
    match = pattern.match(file_name)
    if match:
        head_number = match.group(1)
        df = pd.read_csv(
            file_path,
            encoding='ISO-8859-1',
        )
        df['Head'] = head_number
        df_chi = pd.read_csv(
            f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/¡`•ÿ¶¨§J§¿™R/{file_name}',
            encoding='utf-8',
            usecols=['分目 (編號)', '分目']
        )
        df = pd.concat([df, df_chi], axis=1)
        revenue_by_head_dfs.append(df)
revenue_by_subhead_item = pd.concat(revenue_by_head_dfs)[['Head', 'Sub-head (Code)', 'Sub-head', '分目 (編號)', '分目', 'Estimate 2025¡V26\n$¡¦000', ]]
del df, revenue_by_head_dfs, revenue_by_head_files




# Head
head_description = pd.read_csv(f'{DIR}/head_description.csv')
head_category = pd.read_csv(f'{DIR}/head_category.csv', usecols=[0, 4])
category_eng_to_chi = {'Social Welfare': '社會福利', 'Education': '教育', 'Health': '醫療', 'Security': '安全', 'Infrastructure': '基建', 'Economics': '經濟', 'Environment & Food': '環境與食物', 'Others': '其他'}

# Expenditure by Head and Financial Provision
exp_fin_provision_ = pd.read_csv(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/expenditure-analysis/fin_provision.csv', encoding= 'ISO-8859-1')
exp_fin_provision_chi = pd.read_csv(f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/∂{"}"}§‰§¿™R/fin_provision.csv', encoding='utf-8')


# Expenditure by Head and Expenses
exp_es_ = pd.read_csv(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/expenditure-analysis/es.csv', encoding= 'ISO-8859-1')
exp_es_chi = pd.read_csv(f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/∂{"}"}§‰§¿™R/es.csv', encoding='utf-8')
exp_es_comp = pd.read_csv(f'{DIR}/exp_es_comp.csv')


## detailed expenses under 000
exp_des_ = pd.read_csv(f'{DIR}/estimates-2025-eng/volume1-general-revenue-account/expenditure-analysis/des.csv', encoding= 'ISO-8859-1')
exp_des_chi = pd.read_csv(f'{DIR}/estimates-2025-chi/®˜§@-¨F©≤§@ØÎ¶¨§J±b•ÿ/∂{"}"}§‰§¿™R/des.csv', encoding='utf-8')

# Other spending
exp_other = pd.DataFrame(
    columns=['Target', 'exp_25to26_billions'],
    data=[
        ['Infrastructure 基建', 151.1],
        ['Innovation and Technology Fund 創新及科技基金', 12.4],
        ['Others 其他', 13.1],
        ['Government Bonds - Interest & Repayment 政府債券-利息及還款', 67.1]
    ]
)



# %% Data Cleaning
revenue_sum_cols = pd.concat(
    [revenue_sum_.iloc[:,0:3].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    revenue_sum_chi.iloc[:, 0:3]],
    axis=1
)
revenue_sum_data = revenue_sum_.iloc[:,-2].rename("rev_25to26_millions")
revenue_sum = pd.concat([revenue_sum_cols, revenue_sum_data], axis=1)
del revenue_sum_, revenue_sum_chi, revenue_sum_cols, revenue_sum_data

revenue_internal_cols = pd.concat(
    [revenue_internal_wcat.iloc[:,0:4].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    revenue_internal_chi.iloc[:, [1, 3]]],
    axis=1
)

revenue_internal_data = revenue_internal_wcat.iloc[:,7].rename("rev_25to26_thousands")
revenue_internal = pd.concat([revenue_internal_cols, revenue_internal_data], axis=1).assign(rev_25to26_millions = lambda x: x.rev_25to26_thousands/1e3)
del revenue_internal_wcat, revenue_internal_chi, revenue_internal_cols, revenue_internal_data

# rename the 6th columnin revenue_by_head to 'rev_25to26_thousands'
revenue_by_subhead_item = revenue_by_subhead_item.rename(columns={'Estimate 2025¡V26\n$¡¦000': 'rev_25to26_thousands'})
# for columns 'Head', 'Sub-head (Code)', 'Sub-head'] rename with .rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
revenue_by_subhead_item = pd.concat(
    [revenue_by_subhead_item.iloc[:,0:3].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    revenue_by_subhead_item.iloc[:, 3:]],
    axis=1
)



head_info = head_description.merge(head_category, on='Head', how='left')
del head_description, head_category

exp_fin_provision_cols = pd.concat(
    [exp_fin_provision_.iloc[:,0:4].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    exp_fin_provision_chi.iloc[:, [2,3]]],
    axis=1
)

exp_fin_provision_data = exp_fin_provision_.iloc[:,-1].rename("exp_25to26_millions")
exp_fin_provision = pd.concat([exp_fin_provision_cols, exp_fin_provision_data], axis=1)
del exp_fin_provision_, exp_fin_provision_chi, exp_fin_provision_cols, exp_fin_provision_data


exp_es_cols = pd.concat(
    [exp_es_.iloc[:,0:5].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    exp_es_chi.iloc[:, [1,2,4]]],
    axis=1
)

exp_es_data = exp_es_.iloc[:,-1].rename('exp_25to26_thousands')
exp_es = pd.concat([exp_es_cols, exp_es_data], axis=1).assign(exp_25to26_millions = lambda x: x['exp_25to26_thousands']/1e3)
del exp_es_, exp_es_chi, exp_es_cols, exp_es_data

exp_des_cols = pd.concat(
    [exp_des_.iloc[:,0:3].rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1),
    exp_des_chi.iloc[:, [1,2]]],
    axis=1
)

exp_des_data = exp_des_.iloc[:,-1].rename('exp_25to26_thousands')
exp_des = pd.concat([exp_des_cols, exp_des_data], axis=1).assign(exp_25to26_millions = lambda x: x['exp_25to26_thousands']/1e3)
del exp_des_, exp_des_chi, exp_des_cols, exp_des_data

exp_es_comp = exp_es_comp.rename(lambda x: re.sub(r'[^a-zA-Z]', '', x),axis=1)


# %% Process Data

# Revenue
revenue_sum_cat = pd.DataFrame(
    columns=['RevenueCategory', '收入項目'],
    data=[
        ['Internal Revenue', '內部稅收'],
        ['Internal Revenue', '內部稅收'],
        ['Internal Revenue', '內部稅收'],
        ['Internal Revenue', '內部稅收'],
        ['Other Operating Revenue', '其他經營收入'],
        ['Other Operating Revenue', '其他經營收入'],
        ['Other Operating Revenue', '其他經營收入'],
        ['Other Operating Revenue', '其他經營收入'],
        ['Other Operating Revenue', '其他經營收入'],
        ['Capital Revenue', '非經營收入'],
        ['Transfers from Funds', '從各基金轉撥的款項']
    ])

revenue_sum_ = pd.concat([revenue_sum, revenue_sum_cat], axis=1)

revenue_internal_cat = pd.DataFrame(
    columns=['RevenueCategory', '收入項目'],
    data=[
        ['Other Internal Revenue', '其他內部稅收'],
        ['Profit Tax', '利得稅'],
        ['Other Internal Revenue', '其他內部稅收'],
        ['Property Tax', '物業稅'],
        ['Salaries Tax', '薪俸稅'],
        ['Other Internal Revenue', '其他內部稅收'],
        ['Other Internal Revenue', '其他內部稅收'],
        ['Stamp Duty', '印花稅'],
        ['Other Internal Revenue', '其他內部稅收'],

    ]
)
revenue_internal_wcat = pd.concat([revenue_internal, revenue_internal_cat], axis=1)


# Other Revenue by Sub head
revenue_head_desc = pd.DataFrame(
    {
        "Head": ["001", "002", "003", "004", "005", "006", "007", "009", "010", "011"],
        "HeadDescription": ["Duties", "General Rates", "Internal Revenue", "Motor Vehicle Taxes", "Fines, Forfeitures and Penalties", "Royalties and Concessions", "Properties and Investments", "Loans, Reimbursements, Contributions and Other Receipts", "Utilities", "Fees and Charges"],
        "總目": ["應課稅品稅項", "一般差餉", "內部稅收", "車輛稅", "罰款、沒收及罰金", "專利稅及特權稅", "物業及投資", "貸款、償款、供款及其他收入", "公用事業", "各項收費"]

    }
)
revenue_by_subhead = (revenue_by_subhead_item
    .assign(rev_25to26_millions = lambda x: x.rev_25to26_thousands/1e3)
    .groupby(['Head', 'Subhead', '分目'])
    .agg({'rev_25to26_millions':'sum'})
    .reset_index()
    .sort_values('rev_25to26_millions', ascending=False)
    .merge(revenue_head_desc, on='Head', how='left')
    .assign(rev_25to26_billions = lambda x: x.rev_25to26_millions/1e3)
 )


# Combine
revenue = (pd.concat([
    revenue_sum_.query('RevenueCategory != "Internal Revenue"'),
    revenue_internal_wcat, revenue_other
    ])
    .groupby(['RevenueCategory', '收入項目'])
    .agg({'rev_25to26_millions': 'sum'})
    .reset_index()
    .assign( rev_25to26_billions = lambda x: round(x['rev_25to26_millions']/1000, 1))
    # One-off transfer from other funds
    .assign( rev_25to26_billions = lambda x: np.where(x.RevenueCategory == 'Transfers from Funds', x.rev_25to26_billions + revenue_by_subhead.query('Subhead == "One-off transfer from other funds"').rev_25to26_billions.iloc[0], x.rev_25to26_billions))
    .assign( rev_25to26_billions = lambda x: np.where(x.RevenueCategory == 'Other Operating Revenue', x.rev_25to26_billions - revenue_by_subhead.query('Subhead == "One-off transfer from other funds"').rev_25to26_billions.iloc[0], x.rev_25to26_billions))
    .assign(RevenueCategory = lambda x: np.where(x.RevenueCategory == 'Transfers from Funds', 'ALL Transfers from Funds', x.RevenueCategory))
    .assign(收入項目 = lambda x: np.where(x.RevenueCategory == 'ALL Transfers from Funds', '所有從基金轉撥的款項', x['收入項目']))
    .pipe(
        lambda x: pd.concat([
            x, 
            pd.DataFrame(
                columns=['RevenueCategory', '收入項目', 'rev_25to26_billions'],
                data=[['General Rates', '一般差餉', revenue_by_subhead.query('Subhead == "General Rates"').rev_25to26_billions.iloc[0]]])]
    ))
    .assign( rev_25to26_billions = lambda x: np.where(x.RevenueCategory == 'Other Operating Revenue', x.rev_25to26_billions - revenue_by_subhead.query('Subhead == "General Rates"').rev_25to26_billions.iloc[0], x.rev_25to26_billions))
    .assign(RevenueCategory = lambda x: pd.Categorical(x['RevenueCategory'], ordered=True, categories=['Profit Tax', 'Salaries Tax', 'Stamp Duty', 'Property Tax', 'Other Internal Revenue', 'General Rates', 'Other Operating Revenue', 'Capital Revenue', 'ALL Transfers from Funds', 'Government Bond', 'Deficit']))
    .sort_values('RevenueCategory')
)






# Expenditure by Head and Category
exp_by_head = (exp_fin_provision
    .groupby(['Head', '綱領'])
    .agg({'exp_25to26_millions': 'sum'})
    .reset_index()
)

exp_category = (exp_by_head.merge(head_info, on='Head', how='left')
    .groupby(['Category'])
    .agg({'exp_25to26_millions': 'sum'})
    .reset_index()
    .assign(分類 = lambda x: x.Category.map(category_eng_to_chi))
    .assign(Category = lambda x: pd.Categorical(x['Category'], ordered=True, categories=['Social Welfare', 'Health', 'Education', 'Security', 'Infrastructure', 'Economics', 'Environment & Food', 'Others']))
    .sort_values('Category')
    .assign(exp_25to26_billions = lambda x: x.exp_25to26_millions/1e3)
)


# Expenditure by Component
exp_by_category_n_component = (pd.concat([
        (exp_es
            .query('SubheadCode != "000"')
            .reset_index(drop=True)
            .assign(ExpenditureComponent = exp_es_comp['ExpenditureComponent'].reset_index(drop=True))
            [['Head', 'ExpenditureComponent', 'exp_25to26_millions']]
        ),
        (exp_des
            [['Head', 'ExpenditureComponent', 'exp_25to26_millions']]
            .reset_index(drop=True)
        )],
        axis=0
    )
    .merge(head_info, on='Head')
    .assign(分類 = lambda x: x.Category.map(category_eng_to_chi))
    .groupby(['Category', '分類', 'ExpenditureComponent'])
    .agg({'exp_25to26_millions':'sum'})
    .reset_index()
    .assign(開支組成項目 = lambda x: x.ExpenditureComponent.map(exp_des.set_index('ExpenditureComponent')['開支組成項目'].drop_duplicates().to_dict()))
    .assign(開支組成項目 = lambda x: np.where(x.ExpenditureComponent == 'Innovation and Technology Fund','創新及科技基金', x['開支組成項目']))
    .assign(ExpenditureComponent = lambda x: pd.Categorical(x['ExpenditureComponent'], ordered=True, categories=['Subventions', 'Departmental Expenses', 'Personal Emoluments', 'Personnel Related Expenses', 'Other Charges', 'Innovation and Technology Fund']))
    .assign(Category = lambda x: pd.Categorical(x['Category'], ordered=True, categories=['Social Welfare', 'Health', 'Education', 'Security', 'Infrastructure', 'Economics', 'Environment & Food', 'Others']))
    .sort_values(['ExpenditureComponent', 'Category'])
    .assign(exp_25to26_billions = lambda x: x.exp_25to26_millions/1e3)
)





# %% Sankey Data

# First Layer RevenueSources to Revenue
revenue_sankey = (revenue
    .assign(Item = lambda x: x['RevenueCategory'].astype('str')+ ' ' + x['收入項目'])
    .assign(Source = lambda x: x.Item)
    .assign(Target = 'Revenue 收入')
    .assign(Layer = 'RevenueSources')
    .assign(Value = lambda x: x.rev_25to26_billions)
    [['Source', 'Target', 'Value', 'Layer']]
 )

# Second Layer Budget to Exp Category
exp_category_sankey = (exp_category
    .assign(Item = lambda x: x['Category'].astype('str') + ' ' + x['分類'])
    .assign(Source = 'Revenue 收入')
    .assign(Target = lambda x: x['Item'])
    .assign(Layer = 'Revenue')
    .assign(Value = lambda x: x.exp_25to26_billions)
    [['Source', 'Target', 'Value', 'Layer']]
)

exp_category_other_sankey = (
    exp_other
    .assign(Source = 'Revenue 收入')    
    .assign(Layer = 'Revenue')    
    .assign(Value = lambda x: x.exp_25to26_billions)
    [['Source', 'Target', 'Value', 'Layer']]
)



# Third Layer Expenditure Category to Exp Component
exp_by_category_n_component_sankey = (exp_by_category_n_component
    .assign(Source = lambda x: x['Category'].astype('str') + ' ' + x['分類'])
    .assign(Target = lambda x: x['ExpenditureComponent'].astype('str') + ' ' + x['開支組成項目'])
    .assign(Layer = 'ExpenditureCategory')
    .assign(Value = lambda x: x.exp_25to26_billions)
    [['Source', 'Target', 'Value', 'Layer']]
 )


exp_by_category_n_component_other_sankey = pd.DataFrame(
    columns=['Source', 'Target', 'Value', 'Layer'],
    data=[
        ['Infrastructure 基建', 'Capital Investment Fund 資本投資基金', 151.1, 'ExpenditureCategory'],
        ['Others 其他', 'Other Charges 其他費用', 13.1, 'ExpenditureCategory'],
    ]
)



sankey_data = pd.concat([
    revenue_sankey,exp_category_sankey, exp_category_other_sankey, exp_by_category_n_component_sankey, exp_by_category_n_component_other_sankey
], axis=0)


sankey_nodes = [
    'Profit Tax 利得稅', 'Salaries Tax 薪俸稅', 'Stamp Duty 印花稅', 'Property Tax 物業稅', 'Other Internal Revenue 其他內部稅收', 'General Rates 一般差餉', 'Other Operating Revenue 其他經營收入',
    'Capital Revenue 非經營收入', 'ALL Transfers from Funds 所有從基金轉撥的款項', 'Government Bond 政府債券', 'Deficit 赤字',
    'Revenue 收入',
    'Social Welfare 社會福利', 'Infrastructure 基建', 'Health 醫療', 'Education 教育', 'Security 安全', 'Economics 經濟', 'Environment & Food 環境與食物', 'Others 其他', 
    'Subventions 資助金', 'Personal Emoluments 個人薪酬', 'Departmental Expenses 部門開支', 'Personnel Related Expenses 與員工有關連的開支', 'Other Charges 其他費用',
    'Capital Investment Fund 資本投資基金', 'Innovation and Technology Fund 創新及科技基金', 'Government Bonds - Interest & Repayment 政府債券-利息及還款', 
]


sankey_data = (
    sankey_data

    .assign(Target=lambda x: pd.Categorical(x['Target'], ordered=True, categories=sankey_nodes))
    .assign(Source = lambda x: pd.Categorical(x['Source'], ordered=True, categories=sankey_nodes))
    .sort_values(['Target', 'Source'], ascending=False)
    .assign(Value = lambda x: round(x['Value']/x.query('Target == "Revenue 收入"').Value.sum()*100, 1))
    # rounding error
    .assign(Value=lambda x: 
            np.where(
                x['Source'] == "Deficit 赤字",
                x['Value'] + (x.query('Target == "Revenue 收入"').Value.sum() - x.query('Source == "Revenue 收入"').Value.sum()),
                x['Value']
            )
            )
)






 
# %% Sankey Plot HV


sankey = hv.Sankey(sankey_data, label = "HK Budget 2025 - For Every $100 of Revenue, How Is It Collected and Spent?", )
sankey = sankey.opts(
    label_position='left',
    node_color='index',
    edge_color='Source',
    edge_cmap='Category20',
    edge_line_width=1,
    edge_alpha = 0.5,
    node_line_color='black',
    node_line_width=2,
    node_alpha=0.3,
    
)

sankey
renderer = Store.renderers['bokeh']
plot = renderer.get_plot(sankey)


offset = +150
num_nodes = len(plot.handles['text_1_source'].data['x'])
plot.handles['text_1_source'].data['x_offset'] = [0]* num_nodes
num_right_nodes = 8
left_nodes_selection = slice(num_nodes-num_right_nodes, num_nodes)
plot.handles['text_1_source'].data['x_offset'][left_nodes_selection] = [offset]* num_right_nodes
plot.handles['text_1_glyph'].x_offset = {'field': 'x_offset' }
plot.handles['plot'].x_range.start += (2*offset)




hv.ipython.notebook_extension('bokeh')
data, metadata = hv.ipython.display_hooks.render(plot, fmt='svg')
hv.ipython.display(hv.ipython.HTML(data["text/html"]))

hv.save(plot, f'{DIR}/sankey.html')


# %%
#%% Treemap of Revenue

revenue_treemap_data = (revenue_by_subhead
    .assign(Head = lambda x: x.HeadDescription + '<br>' + x.總目)
    .assign(Subhead = lambda x: x['Subhead'] + '<br>' + x['分目'])

 )


fig = px.treemap(revenue_treemap_data, path=[px.Constant("Revenue 收入"), 'Head', 'Subhead'], values='rev_25to26_billions',
                  color_continuous_scale='RdBu',

)
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25),    title={
        'text': 'Hong Kong Budget Revenue 2025 &nbsp; 香港2025年度財政預算收入 <br>(excluding government bonds and deficit &nbsp; 不包括政府債券和赤字)<br> ',
        'font': {'size': 12}
    })

fig.show()
# save to html
fig.write_html(f'{DIR}/revenue_treemap.html')
