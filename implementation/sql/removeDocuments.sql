-- Remover todos os documentos e chunks do banco de dados
-- Este script limpa todos os dados sem remover o schema/tabelas
-- Útil para resetar a base de conhecimento antes de re-ingerir documentos

-- ==============================================================================
-- REMOVER TODOS OS DOCUMENTOS E CHUNKS
-- ==============================================================================

-- Deletar todos os chunks primeiro (restrição de chave estrangeira)
-- Isso irá cascatear automaticamente se você tiver ON DELETE CASCADE
DELETE FROM chunks;

-- Deletar todos os documentos
DELETE FROM documents;

-- ==============================================================================
-- RESETAR SEQUÊNCIAS (Opcional - garante IDs limpos se você estivesse usando sequências)
-- ==============================================================================
-- Nota: Estamos usando UUID, então não há sequências para resetar
-- Se você estivesse usando IDs SERIAL, resetaria as sequências aqui

-- ==============================================================================
-- VACUUM (Opcional - recuperar espaço e atualizar estatísticas)
-- ==============================================================================
-- Descomente as linhas abaixo se quiser recuperar espaço em disco e atualizar stats
-- Isso é recomendado após deletar grandes quantidades de dados

-- VACUUM ANALYZE chunks;
-- VACUUM ANALYZE documents;

-- ==============================================================================
-- VERIFICAÇÃO
-- ==============================================================================
-- Verificar se as tabelas estão vazias
SELECT 'Documentos restantes:' as status, COUNT(*) as count FROM documents
UNION ALL
SELECT 'Chunks restantes:' as status, COUNT(*) as count FROM chunks;

-- ==============================================================================
-- RESUMO
-- ==============================================================================
-- Este script:
-- 1. Deletou todos os chunks
-- 2. Deletou todos os documentos
-- 3. Verificou que as tabelas estão vazias
--
-- O schema (tabelas, índices, funções) permanece intacto.
-- Você pode agora re-executar o pipeline de ingestão para adicionar novos documentos.
--
-- Uso:
--   psql $DATABASE_URL < sql/removeDocuments.sql
--
--   Ou do psql:
--   \i sql/removeDocuments.sql
-- ==============================================================================
