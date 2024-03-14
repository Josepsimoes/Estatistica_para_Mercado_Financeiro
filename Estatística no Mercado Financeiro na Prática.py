#!/usr/bin/env python
# coding: utf-8

# ---
# # **Estatística no Mercado Financeiro na Prática**
# ---
# 
# Aplicaremos de forma prática e com ativos financeiros reais os seguintes conceitos de estatística:
# 
# - Histograma e distribuições
# 
# - Boxplot
# 
# - Quantile-Quantile Plot (Q-Q plot)
# 
# - Skewness
# 
# - Kurtosis
# 
# - Manipulação de dados de Preço e Dividendos
# 
# - Correlação
# 
# - Spread entre ações com base do Desvio Padrão
# 

# # *Instalações*

# In[2]:


get_ipython().system('pip install sweetviz')
get_ipython().system('pip install vectorbt')
get_ipython().system('pip install yfinance')


# # *Importações*

# In[124]:


import pandas as pd
import vectorbt as vbt
import yfinance as yf
import sweetviz as sv
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pylab
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt


# # *Checando medidas de posição de dispersão nos dados*
# 
# <br>
# 
# 
# Iremos verificar como as metricas se comportam em dados de ativos reais.
# 
# <br>
# 
# Estudos de caso:
#     
# 1. Preços BPAC3
# 2. Preços e retornos do IBOVESPA
# 3. Dividendos
# 4. Arbitragem em ações Preferenciais e Ordinária

# # *Preços BPAC3*
# 
# <br>
# 
# - A extrção será feita através da biblioteca do Yahoo Finance (Yfinance)

# In[47]:


btg_data = yf.download(('BPAC3' + '.SA'), period = 'max')['Adj Close']


# In[48]:


btg_data.head()


# In[49]:


btg_data.index


# Alterando indice para datetime UTC

# In[50]:


btg_data.index = pd.to_datetime(btg_data.index, utc = True)


# In[51]:


btg_data = pd.DataFrame(btg_data)


# In[53]:


btg_data['Retorno diário'] = btg_data['Adj Close'].pct_change()


# In[55]:


btg_data.dropna(inplace = True)


# In[66]:


btg_data.tail()


# # Agora vamos criar um histograma com os dados de fechamento diário do BTG

# In[32]:


# Gerando valores do histograma
go.Histogram(x=btg_data)


# In[77]:


# Criando uma figura

fig_01 = make_subplots(rows = 1, cols = 2)
fig_01.add_trace(go.Histogram(x=btg_data['Adj Close']), row=1, col=1)
fig_01.add_vline(x=np.mean(btg_data['Adj Close']), line_width=3, line_color='blue', row=1,col=1)
fig_01.add_vline(x=np.median(btg_data['Adj Close']), line_width=3, line_dash='dash', line_color='green',row=1, col=1)

fig_01.add_trace(go.Box(y=btg_data['Adj Close'], boxpoints='all', boxmean='sd',),row=1,col=2)

fig_01.update_layout(title_text = 'BTG', width=600, height=300, template = 'simple_white',
                    paper_bgcolor='#f7f8fa', margin = dict(l=20,r=20,t=20,b=20),
                    showlegend=True)


# # *Preços e Retornos IBOVESPA*
# 

# In[93]:


ibov = yf.download('^BVSP', period ='max')['Close']


# In[94]:


ibov


# In[95]:


ibov.index = pd.to_datetime(ibov.index, utc = True)


# In[96]:


ibov


# In[98]:


ibov = pd.DataFrame(ibov)


# In[100]:


ibov['Returns'] = ibov['Close'].pct_change()


# In[101]:


ibov


# In[102]:


ibov.dropna(inplace=True)


# In[103]:


ibov


# # *Plotando Histograma dos preços de fechamento do IBOVESPA*

# In[107]:


fig_02 = px.histogram(ibov['Close'], color_discrete_sequence=['lightseagreen'])

fig_02.add_vline(x=np.mean(ibov['Close']), line_width=3, line_color='red')
fig_02.add_vline(x=np.median(ibov['Close']), line_width=3, line_dash='dash', line_color='red')

fig_02.update_layout(width=600, height=400, template = 'simple_white',
                    paper_bgcolor = '#f7f8fa', margin=dict(l=20, r=20, t=20, b=20),
                    showlegend= True)


# *Acima podemos ver que os preços de fechamento do IBOVESPA apresentam uma Distribuição trimodal, com três picos.*

# # *Quantile-Quantile Plot (Q-Q plot) preços de fechamento IBOVESPA*
# 
# <BR>
#     
# - *Q-Q plot é uma ferramenta visual poderosa para avaliar a adequação de uma distribuição teórica aos seus dados observados. Ele fornece uma maneira intuitiva de detectar desvios da distribuição esperada e pode ajudar na escolha do modelo estatístico apropriado.*
#     
# <BR> 
# 
# - **Alinhamento com a linha diagonal:** *Se os pontos no gráfico estiverem aproximadamente alinhados com a linha diagonal, isso sugere que os dados observados seguem de perto a distribuição teórica escolhida. Isso indica uma boa concordância entre os seus dados e a distribuição de referência.*
# 
# <BR>
#    
# - **Desvio da linha diagonal:** *Se os pontos divergirem da linha diagonal de várias maneiras, isso pode indicar que os seus dados não seguem a distribuição teórica escolhida.*

# In[112]:


stats.probplot(ibov['Close'], dist='norm', plot=pylab)
pylab.show()


# # *Plotando Histograma dos retornos diários do IBOVESPA*

# In[108]:


fig_02 = px.histogram(ibov['Returns'], color_discrete_sequence=['lightseagreen'])

fig_02.add_vline(x=np.mean(ibov['Returns']), line_width=3, line_color='red')
fig_02.add_vline(x=np.median(ibov['Returns']), line_width=3, line_dash='dash', line_color='red')

fig_02.update_layout(width=600, height=400, template = 'simple_white',
                    paper_bgcolor = '#f7f8fa', margin=dict(l=20, r=20, t=20, b=20),
                    showlegend= True)


# Acima podemos ver que os preços de retorno diário do IBOVESPA apresentam oque parece ser uma Distribuição Normal, tendendo um pouco acima de zero.

# # *Quantile-Quantile Plot (Q-Q plot) Retornos diários do IBOVESPA*
# 
# <BR>
#     
# - *Q-Q plot é uma ferramenta visual poderosa para avaliar a adequação de uma distribuição teórica aos seus dados observados. Ele fornece uma maneira intuitiva de detectar desvios da distribuição esperada e pode ajudar na escolha do modelo estatístico apropriado.*
#     
# <BR> 
# 
# - **Alinhamento com a linha diagonal:** *Se os pontos no gráfico estiverem aproximadamente alinhados com a linha diagonal, isso sugere que os dados observados seguem de perto a distribuição teórica escolhida. Isso indica uma boa concordância entre os seus dados e a distribuição de referência.*
# 
# <BR>
#    
# - **Desvio da linha diagonal:** *Se os pontos divergirem da linha diagonal de várias maneiras, isso pode indicar que os seus dados não seguem a distribuição teórica escolhida.*

# In[114]:


stats.probplot(ibov['Returns'], dist='norm', plot=pylab)
pylab.show()


# ---
# # Skewness e Kurtosis
# 
# Enquanto a Skewness se concentra na propagação (caudas) da distribuição normal, a Kurtosis se concentra mais na altura. Ele nos diz quão alta ou plana é nossa distribuição normal (ou semelhante à normal) .
# 
# <br>
# 
# **Distribuições:**
# 
# <br>
# 
# ![image.png](attachment:image.png)
# 

# ### Skewness
# 
# ![image.png](attachment:image.png)
# 
# <br>
# 
# **Skewness (Assimetria):**
# 
# - *A skewness é uma medida que descreve a assimetria da distribuição de dados.*
# 
# <br>
# 
# - *Em uma distribuição simétrica, os valores são distribuídos igualmente em ambos os lados da média, e a skewness é próxima de zero.*
# 
# <br>
# 
# - *Quando a distribuição é deslocada para a direita (valores mais altos concentrados à esquerda da média), dizemos que é uma skewness positiva.*
# 
# <br>
# 
# - *Quando a distribuição é deslocada para a esquerda (valores mais altos concentrados à direita da média), dizemos que é uma skewness negativa.*

# In[115]:


# Skewness Fechamento IBOVESPA
stats.skew(ibov['Close'])


#  A Skewness da distribuição dos dados Fechamento do IBOV acima indica uma assimetria positiva. 

# In[131]:


# Skewness Retornos IBOVESPA
stats.skew(ibov['Returns'])


# A Skewness da distribuição dos dados Retorno do IBOV acima também indicam uma assimetria positiva.

# ###  Kurtosis
# 
# ![image.png](attachment:image.png)
# 
# <br>
# 
# **Kurtosis (Curtose):**
# 
# - *A kurtosis é uma medida que descreve o achatamento ou a forma das caudas de uma distribuição em relação à distribuição normal.*
# 
# <br>
# 
# - *Uma kurtosis alta indica que a distribuição tem caudas mais pesadas e um pico mais pronunciado do que a distribuição normal (leptocúrtica).*
# 
# <br>
# 
# - *Uma kurtosis baixa indica que a distribuição tem caudas mais leves e um pico menos pronunciado do que a distribuição normal (platicúrtica).*
# 
# <br>
# 
# - *Uma kurtosis próxima de zero indica que a distribuição tem uma forma semelhante à distribuição normal.*

# In[116]:


# Kurtosis Fechamento IBOVESPA
stats.kurtosis(ibov['Close'], fisher=True)


# A kurtosis dos dados de Fechamento do IBOV indicam que ela tem uma curva mais suave e caudas mais leves em comparação com a distribuição normal.

# In[132]:


# Kurtosis Fechamento IBOVESPA
stats.kurtosis(ibov['Returns'], fisher=True)


# A kurtosis dos dados de Retornos do IBOV indicam que ela tem caudas mais pesadas e um pico mais pronunciado do que a distribuição normal.

# #  Skweness e Kusrtosis FECHAMENTO IBOVESPA
# 

# In[130]:


sns.kdeplot(ibov['Close'], color="red")

sns.despine(top=True, right=True, left=True)
plt.xticks([])
plt.yticks([])
plt.ylabel("")
plt.xlabel("")
plt.title("Skewness e Curtose Fechamento IBOV", fontdict=dict(fontsize=10))

# Find the mean, median, mode
mean_price = ibov['Close'].mean()
median_price = ibov['Close'].median()

# Add vertical lines at the position of mean, median, mode
plt.axvline(mean_price, label="Mean")
plt.axvline(median_price, color="black", label="Median")

plt.legend();


# # Skweness e Kusrtosis RETORNOS IBOVESPA

# In[133]:


sns.kdeplot(ibov['Returns'], color="red")

sns.despine(top=True, right=True, left=True)
plt.xticks([])
plt.yticks([])
plt.ylabel("")
plt.xlabel("")
plt.title("Skewness e Curtose Retornos IBOV", fontdict=dict(fontsize=10))

# Find the mean, median, mode
mean_price = ibov['Returns'].mean()
median_price = ibov['Returns'].median()

# Add vertical lines at the position of mean, median, mode
plt.axvline(mean_price, label="Mean")
plt.axvline(median_price, color="black", label="Median")

plt.legend();


# 
# 
# # *Dividendos PEPSICO(PEP)*

# In[180]:


# Defina o ticker da ação para a qual você deseja obter os dados de dividendos
ticker = "PEP"

# Crie um objeto Ticker usando yfinance
pep = yf.Ticker(ticker)

# Obtenha os dados de dividendos
dividendos_pep = pep.dividends


# In[181]:


dividendos_pep = pd.DataFrame(dividendos_pep)


# In[182]:


dividendos_pep


# In[183]:


# Calculando a soma de dividendos distribuidos
dividendos_pep['soma_divdendos'] = dividendos_pep.Dividends.rolling('365D').sum()


# In[186]:


dividendos_pep.head()


# In[189]:


dividendos_pep.tail()


# In[185]:


dividendos_pep.soma_divdendos.plot();


# In[204]:


dividendos_pep.reset_index(inplace=True)


# In[205]:


dividendos_pep


# In[197]:


# Coletando dados de fechamento da Pepsico
pep_close = yf.download('PEP', start = '1972-06-05', end = '2024-03-1')['Close']


# In[207]:


pep_close = pd.DataFrame(pep_close)


# In[208]:


pep_close.reset_index(inplace=True)


# In[211]:


# Equalizando os formatos de data e timezone
dividendos_pep['Date'] = dividendos_pep['Date'].dt.tz_localize(None)


# In[212]:


# Juntando os dataframes de Close e Dividendos 
pep_data = pd.merge(pep_close, dividendos_pep, on='Date', how='inner')


# In[214]:


pep_data.set_index('Date', inplace = True)


# In[220]:


# Renomeando colunas
pep_data.rename(columns = {'soma_divdendos':'Soma_Dividendos'}, inplace = True)


# In[222]:


#Calculando Dividend Yield
pep_data['DY'] = pep_data['Soma_Dividendos']/pep_data['Close'] * 100


# In[223]:


pep_data


# In[ ]:


# Removendo dias em que não houve pagamentos de dividendos


# In[224]:


pep_data = pep_data[pep_data['Dividends']!=0]


# In[225]:


pep_data.Dividends.plot();


# In[226]:


pep_data.DY.plot();


# In[227]:


np.mean(pep_data.DY)


# In[228]:


np.median(pep_data.DY)


# In[230]:


sns.distplot(pep_data.DY, hist=True, kde=True);


# In[231]:


stats.probplot(pep_data.DY, dist="norm", plot=pylab)
pylab.show()


# ---
# 
# <br>
# 
# ## *Dividend Yield de PEP foi maior nos períodos onde o preço da ação estava maior?*
# 
# <br>
# 
# 
# ---

# In[233]:


fig_04 = px.scatter(x=pep_data.Close, y=pep_data.DY, width=500)

fig_04.update_layout(width=500, height=500, template = 'simple_white',
                    paper_bgcolor="#f7f8fa", margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False, xaxis_title='<b>Preço PEPSICO (R$)', yaxis_title='<b>DY PEPSICO (%)')

fig_04.show()


# # *Testes de Correlação*
# 
# <br>
# 
# A correlação é uma medida estatística que descreve a relação entre duas variáveis. Ela indica o grau e a direção do relacionamento linear entre duas variáveis.

# ### Correlação de Pearson
# 
# <br>
# 
# ![image.png](attachment:image.png)
# 
# 
# <br>
# 
# A correlação de Pearson, nomeada após Karl Pearson, é uma medida estatística que quantifica o grau e a direção da relação linear entre duas variáveis contínuas. É uma das medidas mais comuns de correlação e é representada pelo coeficiente de correlação de Pearson (r).
# 
# <br>
# 
# - **Se r = 1** isso indica uma correlação positiva perfeita, o que significa que as duas variáveis estão perfeitamente correlacionadas de forma positiva. À medida que uma variável aumenta, a outra também aumenta em proporção constante.
# 
# <br>
# 
# - Se **r = -1** isso indica uma correlação negativa perfeita, o que significa que as duas variáveis estão perfeitamente correlacionadas de forma negativa. À medida que uma variável aumenta, a outra diminui em proporção constante.
# 
# <br>
# 
# - Se **r = 0** isso indica que não há correlação linear entre as duas variáveis.
# 

# In[237]:


corr, p = stats.pearsonr(pep_data.Close, pep_data.DY)
print('==============' * 4)
print('-> Correlação de Pearson: r=%.3f' %corr, 'p=%.3f' %p)
print('==============' * 4)


# ### Correlação de Spearman
# 
# <br>
# 
# ![image.png](attachment:image.png)
# 
# 
# <br>
# 
# 
# A correlação de Spearman é uma medida estatística que avalia a relação entre duas variáveis, sem assumir que as variáveis têm uma distribuição normal ou uma relação linear. Em vez disso, a correlação de Spearman avalia a relação monotônica entre as variáveis, ou seja, se as variáveis tendem a mudar juntas, mas não necessariamente em uma taxa constante.
# 
# <br>
# 
# - Se **ρ=1** isso indica uma correlação positiva perfeita, o que significa que as duas variáveis estão perfeitamente correlacionadas de forma positiva em uma relação monotônica crescente.
# 
# <br>
# 
# - Se **ρ=−1** isso indica uma correlação negativa perfeita, o que significa que as duas variáveis estão perfeitamente correlacionadas de forma negativa em uma relação monotônica decrescente.
# 
# <br>
# 
# - Se **ρ=0** isso indica que não há correlação monotônica entre as duas variáveis.
# 
# <br>
# 
# **Correlação de Spearman é não-paramétrica e não assume distribuição normal**

# In[238]:


corr, p = stats.spearmanr(pep_data.Close, pep_data.DY)
print('==============' * 4)
print('-> Correlação de Spearman: r=%.3f' %corr, 'p=%.3f' %p)
print('==============' * 4)


# In[239]:


# Calculando a matriz de correlação
correlation_matrix = pep_data[['Close', 'DY']].corr()

# Criando o heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Close e DY')
plt.show()


# # *Arbitragem em ações*

# In[241]:


ativos = ['PETR3.SA','PETR4.SA']

spread = yf.download(ativos, period = 'max')['Adj Close']


# In[242]:


spread


# In[243]:


spread.index


# In[244]:


spread.index = pd.to_datetime(spread.index, utc = True)


# In[245]:


spread.index


# In[246]:


# Calculando o Spread entre as duas ações 
spread['ratio'] = round(spread['PETR4.SA']/spread['PETR3.SA'], 3)


# In[247]:


spread.head()


# In[248]:


spread.ratio.plot();


# In[249]:


# Calculando os Spreads com base no desvio padrão 
media_spread = round(np.mean(spread.ratio),3)
um_desvio_min_spread = media_spread - round(np.std(spread.ratio),3)
um_desvio_max_spread = media_spread + round(np.std(spread.ratio),3)
dois_desvios_min_spread = media_spread - (2* (round(np.std(spread.ratio),3)))
dois_desvios_max_spread = media_spread + (2* (round(np.std(spread.ratio),3)))


# In[259]:


fig = px.line(spread, x=spread.index, y=spread.ratio)

fig.add_hline(y=media_spread, line_width=5, line_color="green")
fig.add_hline(y=um_desvio_min_spread, line_width=3, line_dash="dash", line_color="orange")
fig.add_hline(y=um_desvio_max_spread, line_width=3, line_dash="dash", line_color="orange")
fig.add_hline(y=dois_desvios_min_spread, line_width=5, line_dash="dash", line_color="red")
fig.add_hline(y=dois_desvios_max_spread, line_width=5, line_dash="dash", line_color="red")

fig.update_layout(xaxis_rangeslider_visible=False, title_text='Spread (razão) entre preço PETR4 e PETR3 (2000 e mar/2024)',
                  paper_bgcolor="#f7f8fa", margin=dict(l=20, r=20, t=70, b=20),
                  template = 'simple_white',width=900,height=500)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




