# NNSpire Agent (`nnagent`) — Requirements Summary

> **Purpose:** Consolidates ALL requirements for the "NNSpire Agent" pillar (`nnagent/`) as
> documented across the project. This file is created to help the AI assistant understand
> the full scope before the `nnagent` stub is rewritten. Only requirements are listed,
> regardless of completion status.

---

## 1. Identity & Position in the Product

| Attribute | Detail |
|---|---|
| **Pillar name** | NNSpire Agent |
| **Folder** | `nnagent/` |
| **Ontology level** | L6 (application task flow / platform) |
| **Product role** | Third pillar alongside NNSpire Engine and NNSpire Studio |

**Source:** [`TODO.md:11`](TODO.md:11), [`docs/ARCHITECTURE.md:23`](docs/ARCHITECTURE.md:23)

### 1.1 What NNSpire Agent Is

- **Conversational + orchestration agent** — LLM client, NMID test harness, L6 task-flow designer, platform shells
- **Components:** LLM router, NMID renderer, MCP client, Orchestration engine
- **Platform shells:** Win / macOS / iOS / Android / CLI / Web

**Source:** [`docs/ARCHITECTURE.md:23`](docs/ARCHITECTURE.md:23), [`docs/ARCHITECTURE.md:36-38`](docs/ARCHITECTURE.md:36)

---

## 2. Functional requirements

### 2.1 Basic functions

### 2.1 Functions - basic chat
- the agent "chat" I/O window is the main and home /dashboard part of the screen
- THe chat shall feature standart chat layout and functions ((collapsible/expandable, partially expandable, default behaviour configurable) codeblocks/images/other outputs copiable/saveable - also see further in "file parsers"), reporting of the "thinking" (switchable in configuration, if shown, if shown incl. thinking output etc.), error reports (incl. detailed expandable with description what has hapened and been reported by the called provider/runner). The parties (user and AI) should have customisable icons, some pre-default set should be part of the app, configuration in general settings, per provider (both the user and the model icon), per model, per profile.

- The agent shall be able to provide inputs for models inference of these types:
-- standart LLM inference chat text OR standart LLM inference chat audio (spoken/recorded chat input),
--- + atttachemnts of images, audio (file), text/binary files as attachments (see also "parsers"),
-- those inputs mentioned further here and in context of raw neural networks NNSpire Studio values (typically floating points / integeres corresponding to the input layeretc, lists of raw values sets (incl raw tensor vectors), single or multiple correlated n-ths in forms of CSV/text/file) to test the NNs
--- these shall be accessible by a small drop-down icon in the top (right?) corner of the chat input box, where a pop up menu will show the other possibilties apart of standart texct chat (raw input binary, ector, number sets, multiple n-ths of numbers sets etc) extendable by plugins (the default ones shall be also plugins with uniform interface)
- The basic chat should be available also as interactive and batch CLI (the app runnable in terminal/commandline...). There should be some reasonable (standartiesed) way how to invoke various paramters override either on start (batch running) or interactive (hotkey/reserved character/syntax). Note: that is ONLY for those available on the main screen: switching profiles, models, MCP servers, indexing, available KBs,starting automations with parameters, for interactive CLI also listing/switching/Manipulating conversation folders etc.. The cofnguration shall be then done in the config files directly. The app UI with prompt should recognise and use these hotkeys/reserved characters/commandline parameters switches as well (for example in commandline "> nnagent.exe "Model promopt" --model:MyOpenAIConnection/ChatGPT" ... in interactive promopt in CLI or in UI promopt: "$--mode:MyOpenAIConnection/ChatGPT$" instead of mouse click switching in the dropdown). When used in the main UI it should feature intellisense hints and when in the text, it should immpediately carry out the switch iun the UI (for example select a different model in the dropdown).

### 2.1.1 Functions - basic chat - lists
- Each chat context (stored chat as theme) shoud be directly recallable from a list on the main UI and one can continue there. The chat themes may be organised in "folders" (category folders in the list, even multi-level). The caption should be renameable (both of the chat as well of the folder, the caption of the chat should be created ad-hoc based on the first promopt as an aditionall LLM call, yet this should be cofnigurable (switchable off, then the chats would be named simply by keyword "Untitled" and date and time))
- each of the folders as well single chats as well the entire list should be made exportabve/improtable/backupable (and show where theys respective data files - each one for each file + folders and list index) are stored
- each of the chats should be displayble in various extent of "raw" forms (chat question-reposnse only (with collapsed, partially expanded, fully expanded code blocks, each of them copiable into clipboard/downloadable(save as file, if a corresponding write parser plugin (see further "parser") is present, then also saveable as.. (for example .md as pdf, docx etc))), chat with thinking blocks collapsed/expanded, chat with MCP calls blocks collapsed/expanded featuring the briwf overview or full raw text of the mcp req and resp, entire chat in raw form graphicalyy enhanced (to show paramter calls, sys promoptps, etc.) or full RAW (nice formatting) or full RAW (no formating) JSON
- the "list" should also feature the switch to display history (calls done, process log, outputs) of runs of various automations (see "automations" further below) as well configured automations to be run/ready
- the list should also allow import from the "ChatGPT" JSON export
- organization of the folders/chat items should allow also ctrl+c/x/v as well drag and drop (both for whoel folders/fodler tress as well for single chat items)
- a folder may be mapped to a physical directory thus overriding the default data storage folder, so theoretically, the user may create a specifici "pseudofolder" structure, that points to different folders in this real data storage space on harddrive, and have an "integrated view" on that in the nnagent client. The export/import of the "root folder" may also feature "export as links only" so that the real data are NOT part of the export, so only the "addresses of folder locations" where the real data exist can be exported. Then, when imported into another client, it is expected that those data exist on the same folder in filesystem locations as originally.
- the export/import may be done also to/from NFS, FTP, SMB/CIFS, WebDAV, SharePOint, OneDrive and/or Google drive, with the feature to be automatically scheduled but only for a given folder (so part of the folders may be locally, part of that scynced from the network, with only pointers to folders or to the entire data). Please note, that these specific settings (network drive source/destination + scheduling) may be multiple (one set for each folder, so you may have a fodler synced locally from a network drive 3 times a day, another one from OneDrive once a day etc.)
- please note, that in case of a multi-user instance of the agent, depending on the permissions set to each of the folders, a user may see also folders from other groups (or from groups he is a member of ... those groups may be also used as "containers", eg. not containing any user, but being used as storage for the chat history and tempaltes/automations/profiles), other specifici users either from that same system or from its other instance (the system will feature an REST-kind API for that and either when opening a folder (importing for sync) one may either reference only the different local user/group or give a full URL to another systems instance endpoint and then browse the groups/user chat folders there). It may also be, that in a single-user system the user will like to have the folders of another API (online) or from another folder on the same HDD (for example parent watching the conversation of their kids on the same PC). Note, that also the such folders do not necessarilly to be "imported" (into the data of the user) as normal import/sync, but only "loaded on demand/expand" from the datasource (hdd folder, NFS, API, One drive... see above) specified. Eg. there may be and a) import (data is read, copy stored locally), b) scheduled import (data is read regularly and stored locally), c) synchronization (data is exported/imported regularly from/into the local storage), d) cached/on the fly read(/write) from/into remote data source. The rest of the principle sahll be allways the same and transparent to the user - being a "folder" with sub-folders/chat items, selecting a chat(conversation) item will display that in the window.
-- every folder, chat(conversation) item as well specific prompt should be addressabel using its full URL (nnagent-local://C:/Users/Username/Data/NNAgent/storage/Folder1... or https://mynnagent.lukas.plachy.eu/API/GroupX/FolderY...) and this link should be user shareable and also openable (simply input into the chat or through the Import item of the folders)

### 2.1.2 Functions - basic chat - controls
- the "New chat" button should feature also a dropdown (or another buttons "Start new task", "Schedule new task(s)") to use any automation form a list of automation templates confgiured for a specific run (the first item should be "edit" to navigate into the automation list editor, see further below)
- there may be also more "New _something_" buttons, for designing an automation template or an inference graph etc.  
- key part of the "chat" input box shall be also items:
-- reserverd character calling/referencing a prompt template
-- dropdown for selecting "model" to be used for the prompt (categorised by providers, those marked as "favourites" on top as first category, each line having two icons/context menus to toggle the "favourite" option and one of the also the "Default" option)
--- if a specific model is selected, the "profiles dropdown" group "model profiles" will feature all profiles where this specific model is used explicitly
--- not that within one chat/conversation, the model MAY be switched! so the entire conversation history of being conducted with the previous model goes to another one
-- dropdown for selecting "profile" (first line to feature "none" item as none profile selected, the second the "Edit" item to navigate into profiles editor, the next group being the "favourite profiles" (those marked by the user as "favourite") and next group shall be "model profiles" if a specific model is selected in "models" dropdown (the defautl one for example) featuring only those profiles featuring that model explicitly) if a profile is selected, the models dropdown will filter out only those models that are supposed to be used within that profile ... for further details see below
--- the dropdown shall also feature the "favourites" and "default" toggle buttons. Pay attention, that the "default" buttons of "models" and "profiles" dropdowns may cancel-out each other (if a default model is selected that cannot be used with the "profile" default item selected, then the later selected takes precedence, if the yre in line, then they work both, for example if a profile (marked as default) enables usage of 4 models and one of them is also selected as "default" then every new chat will start with those)


### 2.2 Functions configuration - basic chat
Note: the layout of the basic chat should be also subject to skinning (not only the colors, fonts, backrounds, images etc., but the entire parts of the main home UI - list, buttons, promopt input, chat output, chat questions, answers...)

### 2.3 Functions - skinnability and UI
- Theme token vocabulary is shared in spirit with the `nnagent` skin descriptor so the two products feel like one family
- Hot-reload: edit a theme file → UI updates without restart (mirrors `nnagent` behavior)
- Align the token vocabulary with the `nnagent` skin descriptor so themes are conceptually portable between the two apps
- `ThemeDescriptor` + built-in themes (ADR-049)
- the skin should be responsive design, in compact screens with a humburer menu featurng the chat folders/conversations, and fixed on th ebottom the controll bttons (grouped a one with dropdown selection arrow) and profile configuration (avatar icon + name/nic navigating to the profile configuration with the ability (popupmenu) of navigating ito the settings odf the app, the profile or profilew switching (see 2.18.1))
**Source:** [`TODO.md:1978-2005`](TODO.md:1978), [`TODO.md:1528`](TODO.md:1528)

### 2.4 FUnctions configuration - skinnability
- There shall be default skins for all the distribution packaging features (see further below in architecture) each at least featuring light, dark, light-dapened (gray), dark-solarized, dark-gray scheme.

### 2.5 Functions - model providers
- The models used in the client shal be callable by standart (LM Studio, OpenAI, ...) APIs, extendable by plugins and configurable (urls, endpoints, authentications, API keys, procols paramters fine-tuning (both on the payload JSON/MCP etc. level as well on the HTTP(s) level) in the configuration settings UI of the agent, of course NOT allowing those functionalitities (raw input for example) that the standart APIs do not provide (raw input shall be only for the NNSpire Studio integration)

### 2.6 Functions configuration - model providers
- The entire app shall have model providers configurable (incl. the pluginable ones, so the plugin should carry its own configuration UI UI component/controll for configuration as well for implementing the required interface/call to perform its main functionality/ies) (see further)
- In future there will also be the option of a local ollama or other inference machine to be run and loading own local models from a GGUF or similar files repository. This is not to be implemented now, but as not for the future, so that this is not blocked somehow by design.

### 2.7 Functions configuration - model settings
- Each of the models (fetchable from the model provider if the API allows to list them) shall have configurable default parameters (temperature, system prompt and all the others that the interface of model provider call allows), these may be overridable (or locked) for user's favourites profiles/specific chat settings (see further) 


### 2.8 Functions - profiles
- There shall be so called "profiles" (associable with one or more modells) selectable within the main chat prompt (the prompt shall allow selection of two main things:
-- a model from the list of "engaged" models from those the providers have provided - including the option to mark "favourite" models to be on the beginning of the list and one of them as "default"
-- AND a profile ... the "profile" will filter by those profile relevant to selected model AND then there will be all the other profiles (separated in the list) that being selected will also force-switch/filter the start of the lsit of models ... then of course the model list will continuee to provide also (separated) the list of OTHER "models", which, if selected will again switch the first part of the list of possible "profiles") the "profiles" may be also marked as "Favourite" (one of them as default))


### 2.9 Functions configuration - profiles
- part of the profile may be also a specific set of "MCPs", "knowledge bases" and "prompt templates" (by default: all configured shall be available, only if required one may disable some of them (in case of MCPs up to the level of tools/actions))

### 2.10 Functions + configuraiton - MCPs & webfetch
- the app should provide information (and configurable URL, authorization, default tool listing ("test button") and tools (disablement, all tools/actions enabled by default once listed from the MCP) selection) for integrated MCP servers that can be called within the chat by the AIs
- there shall be an important deafault MCP part (also as plugin, only distributed with the app by default) of the app for webfetch from adjacent AIAPI project (see neighbourhood folder to NNSpire project), to be able to do the same webfetch and eventually interactive browser (browser helper in AIAPI) controll to browse web and input webpages and/or get its outputs
- files read/write as default (by default switched off!) configurable "Firewall" of "explicitly allowed" and "explicitly denied" paths (incl. wildcards) or filenames, extensions etc., default policy select (default deny or default allow)
- terminal commands as default (by default switched off!) configurable "Firewall" of "explicitly allowed" and "explicitly denied" commands (incl. wildcards), default policy select (default deny or default allow)
- the NNagent client should advertise these to the AI model in input prompt, but the agent will ASK the user on MCP request by the model (same as for exmaple VSC: allow this specific one incl.parameters allways/in this chat session, allow this command/file/folder access with any paramters allways/in this chat session, allow any allways/in this chat session, add detailed configuration.. (calling the settings with prefilled command/filepath+operation to be allowed/denied by the user and customzed further by wildcards etc.))

### 2.11 Functions - prompt tempaltes
- the app should provide a configurable list of prompt templates. A prompt template shall be callable directly in prompt text typing (for example by @ or # keycharacter, although that character must be typable also in any string as its original meaning, so only after the character is typed and a tempalte of that name/filtered name as typing further is explicitly selected with "enter" key or LMB then and only then that tempalte is used in the prompt!).
- A prompt template is simply a pre-defined prompt text (having its title = the identifier used to select from list when the reserved character such as # or @ is typed ... can be no-spaces allowed, only underscores) and and optional note/comment (single line input so that the text does not get to chatty), that will become part of the prompt (graphically it should be visible in the prompt text input as well in chat as expandable/collapsible name/title of that tempalte, but when expanded, it should show the entire prompt that was send over by that template, with a switch showing either the configured text OR the text with the values paramters filled-in)

### 2.12 Functions configuration - prompt tempaltes
-In the template text there should be the option of:
-- putting there placeholders (for exmaple in double double brackets {{}} woth paramterised name, input caption, data type (string, integer, date, time, datetime, single/multi selection from a predefined list, the list also paramterised simply as text in the double-double brackets, on the place of the list also intervals may be configured / regex filters for the non-list values types))
--- intellisense when configuring a placeholder {{}} when typing and using the separation character, so that it say first "name/id,type,[default caption],[list values]" based on what "position" in the {{}} and separators the user has its cursor
-- OR a "switchable" part (for example by *-the_text_that_can_be_Switched_on_and_off-* bullet, when input into the chat promopt edit, all these will be listed as checkbox-bulleted list with checkboxes on or off by default (configurable for example *on- -* or *off- -*)), not that the list may be mutli layered (multi levels) distinguished by the numer of dashes "-", so *-Group 1-*\r\n*--Group 1 Item 1--* etc.);
- prompt templates may be also retrieved from structured files (also configurable/selectable as whole folder of such files), 
-- one or multiple (which creates mutliple groups of prompt templates in the app, the first one being the "default" one of the user configuration, the other from referenced external files/folders when those were slected eg. one user "add" action of folder/file = one additional group, in multiuser environment this can be also - when allowed by the author/owner user - obtained from other users/groups (where the user/owner/admin can also confiure them directly in the app or simply load from a file), this can happen by-default in case of a specific group membership)


### 2.13 Functions - knowledge base
- the app should provide a knowledge base - each "KB record" shall feature configuration whether that KB should be a 
-- "local files ad-hoc one" (=searchable by another built-in MCP - combination of terminal/commandline access and files read)
-- or "indexed embedded RAG".
- This should be made available to the chat promopt calls/model MCP calls etc. (if the model supports that, if not, it will be simply ommited).
- KB being made available to the promopt should be switchable by a specific icon (if not already by a profile) in the main UI below the promopt.
- KB being indexed for RAG should have semarate icon displaying the status of the indexing (errors when some connections fail to the embedding model or the vector db, organge when indexing is in progress and green when indexing idle and fully done)

### 2.14 Functions configuration - knowledge base
- The indexed embedded RAG should require an external vector DB (again, plugins), by default we will ship plugin to contact "Qdrant", the others later.
-- That would also require the configuration of one of the models (as the embedding models),
-- optionally a document parser plugin set (or only text files will be processed by the dafult internal plugin of parser of the agent - for "parsers" see further) and optionally rerank and vision models.
- Note, that the "RAG repository" may be filtered, thus there may be one or more knowledge bases based on the same RAG may have different filters (different collections, different tables etc.); the app should then index the files (first start upon command of the user, then allways automated on start, but in background thread not disturbing user, the behaviour may be switched off, so only when explicitly requested a reindex is started, the controll and switch should be in some status icon in the main UI right in the chat). 
- Important note: a specific type of RAG repository + fileparser combination may be DB-like sources (system registry, PIM databases such as for example Ooutlok .pst or Thunderbird, or even ODBC data sources, MS-SQL/Oracle/PostreSQL connectors to a specific procedure/function/table(s)/views(s) etc. ... of course the "file parser" is in that case the "source" connecting to the PIM, registry whatever complex DB and the RAG is then the DB/whatever holding the index, BUT in the future, those may blend into one, so the system should be aware of and ready for that!)
- The parsing should be schedulable based on a) manual trigger (the icon in main UI should lead to that configuration UI for the KB configurations to allow manually triggering that), b) schedule "on change" (monitoring the filesystem/data source, it it allows), c) scheduled to re-check file changes (polling), d) force-scheduled (allways done fully). b) and c) may be arbitrary combined with on d) or anytime manual lounch of a)
- The parsing embedding should be budgeteable based on maximum a) number of prompts b) number of tokens c) size of data input; there shall be a "budged reset" schedule (on a specific time, date)


### 2.15 Functions - parsers
- the app should by default work only with plaintext files (not by extension, but by content), yet it may be able to attach a file to prompt/index it in KB/use in the built-in MCP file read/Write, using a parser plugin that converts any arbitrary binary file into text input (or another binary input that the model is able to work with, such as in case of images for example).
- The file parser may be read-only (for indexing, prompt input, MCP read) or read/write(when altered by the default built-in MCP write).
- Note, there shall be one specific default fileparser: when an "output as downloadable file" is returned (binary, text/.md, Base64 encoded) by the model as a "block", it should be wrapped (in case of binary/Base64) as a "downloadable file", in case of ".md" or "code" and inlays such as that, the block shall be displayed specifically, but the fileparser for "save" should be manifested only as an option of the display div in the main UI chat messages.
- Important note: a specific type of RAG repository + fileparser combination may be DB-like sources (system registry, PIM databases such as for example Ooutlok .pst or Thunderbird etc. ... of course the "file parser" is in that case the "source" connecting to the PIM, registry whatever complex DB and the RAG is then the DB/whatever holding the index, BUT in the future, those may blend into one, so the system should be aware of and ready for that!)


### 2.16 Functions - automations
- The NNAgent app shall feature an editor and runner of automations.
- This shall work as well in the standalone UI app (when launched) but also as background workers (if the app is a container/service/daemon). 
- The "automation" shall be in fact an "orchestration editor" 
-- (something like when configuring "diffusion" graphs in Comfy, YET in this case the raw NN inputs/outputs are not the primary goal
-- (note: sometimes later this might be also an option when a local model runner is implemented into the app, so it MAY double for things like Comfy, but ATM it is not the aim, only we must not block that way in future)),
- in this case the main "blocks" (nicely graphically displayed, linked by lines representing the input/output transfers, once one block finishes its operation, the output is transferred to another block that executes etc.) shall be of these types:
-- AIprompt
--- (as input: previous output of any block + paramters (may be also output of a block),
--- as oputput: the returned AI model inference output
--- (MAY contain also intemediate self-loops to call MCP, terminal, files, KB and also other AI or script blocks while the inference is running and requiring the tools advertised etc.
---- that self-loop may be configured directly on the block or as an "intermediate" output, for example if a script block is to be called or another AI block so that the AIs can "chat" between them so that that "subloop" (block sub-call) will be visible and explicitly defined in the editor);
-- script (java_or_typescript/npm/python) using the "inputs" (providing them to the source code as variables or something of that kind) and providing "outputs" to the code to be altered, when code finishes, those are carried to the next calleable block);
-- dll/other low-level compiled plugin (the NNagent will specify the interface and ship as part of the distro, so that it has the inputs/outputs/intemediate output-inputs, but its up to the implementator what he wants to within the plugin);
-- timer (having only "empty" output - a line draws to the input of for example an AIprompt block or script block);
-- decission block(in fact a special type of the script block);
-- group block = can contain other blocks and their relations/links/lines, a kind of a "procedure"/"function" (the inputs/outputs then manually specified when editing an linked from the borders or from preset immutable "proxy" blocks, that represent that groupblock's paramaters);
- please note, that each block may have MULTIPLE PARAMETERS as output (YET, if specified, grouped into one message = one line, so from a block, there may run multiple lines, yet each may containe one or more paramter values of outputs/inputs);
- the running of that automation shall be doable also right in the editor (with preset profiles of global parameters! see further here) and visuallised in the editor (highghlighted block) and thoroughly logged in console output in the editor);
- there may be general paramters of any of the block that are pre-set, but they may be filled by placeholders ({{}}) which shall be then the global paramters of the entire automation template (OR of the group block only! yet even a group block or block inside a group block may simply use the entire atuoamtion global parameter), these global parameters might be set by either the timer, or, as one automation my have mutliple instances to be run with different global paramters/timings, by the instantiation of the automation.
- The automation can be then scheduled to be run (if it contains a timer) or simply listed in a list of automations ready-to-run (from UI same as "new chat" then the button while in default allways being "new chat", yet it will feature a small expand arrow to list of avaialble automations and their presets - note that one automation may have multiple instances prepared to be run! so there has to be configurable a kind of automation list of automations ready-to-run (as kind of functions, including global parameters specific values, edited/saved directly at that list but those MAY be copied from the automation tempalted debug global paramter preset(s)) that differs from the automation editor list of automation tempaltes and their debug global paramters presets)
- Yes, when the client UI app connects to a server, which features some non-timed automation runs, then the client when running the automation may either execute it within the service/container/daemon (eg. on the server) OR locally, depending on the default configuration of the automation function ready to be run, or explicit question to the user (of course, if the resources - file parsers, KBs, model provider configurations etc. - required by that automation are not present on that server on on that users client UI app, then the autoamtion will fail, but that is left to the user/autom.tempalte.author discretion).

### 2.17 Functions - SSO users/groups/authorizations
- the app shall feature a SSO with user, groups and authorizations. 
- Groups = sets of users that are assigned specific authorizations (and also linked some pre-defined/allowed model providers (add/edit/use), models to be executed/called (API calls including)(edited for other authroizations = assigning into another groups, edited paramters globally, edited paramters for that users call (does not apply if the prompt profile specifically assigned to that group to be used does change that param), switched on/off generally in the system), prompt templates (usable, editable+deletable, cerateable), parsers/MCPs to be used (calleable, editable globally, editable for the user, disableable), automations to be called/edited/listed in editor/tempaltes instantiated as functions, parsers used/administered, KBs (administered, added locally, usage of RAG sources, editability of RAG sources), parsers (used, administered))
- Settings shall be typically configurable as single-user (when installed as default single-user UI) so that all the hassle of users, groups and authorizations shall be not relevant (will be preconfigured for default access with no login), or as user-space daemon/service (the same, no specific user login for localhost interactive access).
- When installed as system/other account service/daemon/container the user configuration of users, groups and their authorizations shall be fully activated and configurable (same as in single-user service/UI app) but in this case requiring default settings of at leas one administrator account to perform further settings


### 2.18 Functions - APIs
- the app should provide an API access (of the UI to setup/use the service + the automations I/O if configured + the models "proxy" - a kind of model router (the APP, UI or service, shall exhibit an OPenAI and others compatible APIs to serve as the model server))

### 2.18.1 Functions - APIs / environment switching
- there shal be a "profile" icon in topright corner/hamburger menu (see basic chat) where the user profile icon is shown (avatar icon). In single user mode (locl, on disk) only the main configuration file/storage path shall be given + name/nicname of the user + selected icon (some pre-generated icons should be there bundled with the software for the user as wel for the "AI"ů" agent to be displayed then in the chat dialogue texts to distinguish the "speaker" of the text) but this should allow also profile switching, either preconfigured profiles (in configuration) or "LIVE" (on-demand ad-hoc profile switch). Those ad-hoc as well preconfigured ones shall feature: a) local hdd/Storage volume folder (also containing the other configured profiles to be connected to, one of these local hdds/srtorage configs may be the "default" one, especially for single-user app mode) (featuring connection/profile name, folder path/filename), b) aN ODBC or DB connection (especially for the WebUI, but also usable for the app desktop/mobile UI)(feturing connection/profile name, the type (plugins), host and port or ODBC source name, username and login password or API key or oAuth or domain/LDAP auth (for Windows-integrated RDBMs),), c) an API URL of the app service (may be also localhost running service), the API as REST-like API see further below (configuration featuring connection/profile name, URL, HTTP authorization certificate/username/password/API key/server certificate trust (not verified, all trusted, only specifc SN/fingerprint list trusted, OS trusted CA/certificates trusted)). Each of them may be used multiple times to reprsent diffeerent sources. 

### 2.19 Deploy and behaviour
- The UI is the main runtime and deployable part
- But all the functions and configurations (model providers, models, model and chat profiles, MCP & webfetch, prompt tempaltes, knowledge bases + RAG repositories (multiples), files parsers, automations, users, groups, autohrizations, API accesses, -> see their abovementioned functions) shall be deployable also as OS Service (Windows) InitDaemon/SystemDaemon (MacOS/Linux) Container see further below


## 3. Other constraints, platform and architecture
- Platform assembly shall target: Win / macOS / iOS / Android / interactive+batch CLI / Web
- Packaging should be: Standalone app + CLI (all platforms), container with a web UI (and a pseudo-CLI - a different form ot the webui only featuring would-like-to-be CLI) (Web / container platforms), Office UI plugin (MS Office+Libre Office+OpenOffice for Win AND the same for macOS Linux, Libre/Open Office for Linux), Widget/Todaysreen/Floating app (iOS + Android), Service (WindowsService (with installer) in Win, init.d and systemd daemon in macOS/Lin)
- The platform is a key decission to be facilitated by AI model: Qt, modern Elctron+Rust or simmilar smaller node.js app?
- Helping UI frameworks: ???? (especially the blocks-and-lines edit of automations, and the js.scriptiong of automation as well npm/library management for the automations)
- Key requirements: as small as possible (but back-portable as far as possible, pref. to Win XP or even 3.1, macOS 8, early versions of Android, old Linuxes, MS-DOS? ... those are not "hard" targets, only estimations of thresholds)
**Source:** [`docs/ARCHITECTURE.md:38`](docs/ARCHITECTURE.md:38)
- Consistent with ADR-043 (UI portability)
- settings as jsons/XMLs, separate files for portable/separated export/import settings:
-- main settings, model providers, models, profiles - this shal be the root file, usually in the root of the storage folder of that profile of theuser, this is the file that will be read/servedthrough API/from DB/from hdd/storage volume when the user "switches" profile (or the app starts with the single-user default local profile)
-- user, groups and authorizations + API access config (the authorizations should use either a reasonable "GUID" references to other parts that can be exported independently or a full value-keys in case for exmaple model indetifiers in the providers)
-- prompt templates
-- MCPs
-- knowledge base & RAG & parsers settings
-- automations - each automation tempalte (incl debugging global parameters presets) in single file BUT an entire list of automations prepared to be run with specified global paramters (and then this file/list will also include the automation tempaltes in its own)

- when in configuration a value (ANY!) is dependent on an existing data entity/list already "known" in the (other modules) settings of the app, the setting should provide a combobox/selection for that value, but also allow direct text input
- all plugins referred here should have an uniform interface design which shall be puvblishedas aprt of the distro, so that anyone can inherit/implement those interfaces and develop own plugins

## 4. Architectural Boundaries

### 4.1 Relationship to NNSpire Studio

- Agent is callable **from within** Studio via an **Embedded Agent Panel** (Phase 3.5)
- The boundary protocol is the **MCP typed boundary node** (ADR-044)
- Studio exposes an MCP server interface; Agent connects as a client
- Symmetric — Agent can also drive Studio model operations as MCP tools

**Source:** [`docs/ARCHITECTURE.md:49`](docs/ARCHITECTURE.md:49)

### 4.2 Relationship to NNSpire Engine

- Agent reaches Engine through the provider **NNSpire Runner API** (HTTP/IPC)
- Same deployment channel used for production serving
- Agent **never** calls Engine internals directly

**Source:** [`docs/ARCHITECTURE.md:51`](docs/ARCHITECTURE.md:51)

### 4.3 Ontology Position
- The user's "layer 0" (Client / Foundry / Agents) is ontology **L6**
- Lives in the separate `nnagent` project **outside** NNSpire's structural tree
- Appears only as a typed **orchestration-boundary node** (ADR-044)
- Numbering inversion: user-top "0" = ontology-top "L6"; hardware L0 is the floor

**Source:** [`TODO.md:1503-1505`](TODO.md:1503), [`TODO.md:1655-1658`](TODO.md:1655), [`docs/ARCHITECTURE.md:703-706`](docs/ARCHITECTURE.md:703), [`docs/modern_ai_systems_ontology.md:3264-3266`](docs/modern_ai_systems_ontology.md:3264)

### 4.4 Engine Independence

- Agent reaches Engine **only** through Runner API (HTTP/IPC)
- Never calls Engine internals directly
- Same deployment channel as production serving

**Source:** [`docs/ARCHITECTURE.md:51`](docs/ARCHITECTURE.md:51)


## 4.5. Cross-Pillar Integration Seams
| Seam | Agent Deliverable | Unlocks |
|---|---|---|
| **S1 — NMID + Runner** | NMID reader + Runner API client (Phase 9) | Agent headlessly tests any Studio model |
| **S2 — Agent Panel in Studio** | `nnagent-core` packaged as embeddable library | User chats with LLM while editing a model |
| **S3 — MCP boundary node** | Agent connects as MCP client (Phase 9-10) | Agent triggers training / export / inspection |
| **S4 — Orchestration driver** | Agent L6 orchestration engine (Phase 10) | Full bidirectional Studio ↔ Agent control |

**Source:** [`TODO.md:402-407`](TODO.md:402)

## 5. Development Phases & Deliverables
- TODO BY AI

---

## Sources Index

| Source File | Relevant Lines/Sections |
|---|---|
| [`TODO.md`](TODO.md) | :11 (status), :394-427 (parallel tracks + seams), :1503-1505 (L6 ontology), :1528 (theming ref), :1648-1658 (ADR-044), :1978-2005 (theming) |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | :23 (pillar table), :35-40 (architecture diagram), :49-51 (boundary protocols), :703-706 (L6 position) |
| [`docs/modern_ai_systems_ontology.md`](docs/modern_ai_systems_ontology.md) | :3264-3266 (L6 numbering note) |