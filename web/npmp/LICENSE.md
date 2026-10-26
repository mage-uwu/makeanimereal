# web/npmp: what is here and its terms

| file | what | source | terms |
|---|---|---|---|
| `npmp.bin` | MoCapAct's multi-clip policy (full dataset), weights as float16 | [microsoft/mocapact-models](https://huggingface.co/microsoft/mocapact-models), `multiclip_policy.tar.gz` | CDLA-Permissive-2.0 (below) |
| `npmp.json` | its layout and normalizers' place, and where dm_control's observables live in MuJoCo | made here (`mocapact/export_web.py`) | as the repository |
| `cmu.xml` | dm_control's CMU humanoid (2020, position controlled) on a floor, as dm_control compiles it | [google-deepmind/dm_control](https://github.com/google-deepmind/dm_control) | Apache-2.0 |
| `clips/*.json` | eight CMU motion capture clips as dm_control's CMU 2020 data fits them to that humanoid | `cmu_2020_dfe3e9e0.h5` (DeepMind), from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/) | free to use, modify and redistribute, commercial use included ([CMU's terms](http://mocap.cs.cmu.edu/faqs.php)) |

`web/npmp.js` and `mocapact/` follow the architecture of MoCapAct's `NpmpPolicy`
([microsoft/MoCapAct](https://github.com/microsoft/MoCapAct), MIT, Copyright (c) 2022 Microsoft
Corporation) and the observations of dm_control's `MultiClipMocapTracking` (Apache-2.0).

## Community Data License Agreement – Permissive – Version 2.0

This is the Community Data License Agreement – Permissive, Version 2.0 (the “agreement”). Data
Provider(s) and Data Recipient(s) agree as follows:

**1. Provision of the Data**

1.1. A Data Recipient may use, modify, and share the Data made available by Data Provider(s) under
this agreement if that Data Recipient follows the terms of this agreement.

1.2. This agreement does not impose any restriction on a Data Recipient’s use, modification, or
sharing of any portions of the Data that are in the public domain or that may be used, modified, or
shared under any other legal exception or limitation.

**2. Conditions for Sharing Data**

2.1. A Data Recipient may share Data, with or without modifications, so long as the Data Recipient
makes available the text of this agreement with the shared Data.

**3. No Restrictions on Results**

3.1. This agreement does not impose any restriction or obligations with respect to the use,
modification, or sharing of Results.

**4. No Warranty; Limitation of Liability**

4.1. All Data Recipients receive the Data subject to the following terms:

THE DATA IS PROVIDED ON AN “AS IS” BASIS, WITHOUT REPRESENTATIONS, WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED INCLUDING, WITHOUT LIMITATION, ANY WARRANTIES OR CONDITIONS OF
TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

NO DATA PROVIDER SHALL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING WITHOUT LIMITATION LOST PROFITS), HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE DATA OR RESULTS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.

**5. Definitions**

5.1. “Data” means the material received by a Data Recipient under this agreement.

5.2. “Data Provider” means any person who is the source of Data provided under this agreement and in
reliance on a Data Recipient’s agreement to its terms.

5.3. “Data Recipient” means any person who receives Data directly or indirectly from a Data Provider
and agrees to the terms of this agreement.

5.4. “Results” means any outcome obtained by computational analysis of Data, including for example
machine learning models and models’ insights.
